# TimeDependentCallable subtypes must be defined at top level. `_test_TDPipeOpt` declares that
# its per-fold value is an optimiser and records the raw fold data its context carries;
# `_test_TDPipePrevW` additionally declares its previous-weights requirement.
struct _test_TDPipeOpt <: PortfolioOptimisers.TimeDependentOptimiserCallable
    seen::Vector{Any}
end
function (c::_test_TDPipeOpt)(ctx::TimeDependentContext)
    c.seen[ctx.i] = ctx.rd
    return isodd(ctx.i) ? EqualWeighted() : InverseVolatility()
end
struct _test_TDPipePrevW <: PortfolioOptimisers.TimeDependentOptimiserCallable
    seen::Vector{Any}
end
function (c::_test_TDPipePrevW)(ctx::TimeDependentContext)
    c.seen[ctx.i] = ctx.w_prev
    return EqualWeighted()
end
function PortfolioOptimisers.needs_previous_weights(::_test_TDPipePrevW)
    return true
end
@testset "Pipeline prediction and CV" begin
    using Test, PortfolioOptimisers, TimeSeries, Dates, StableRNGs, Statistics, FLoops

    # business days only: an irregular calendar for date-based splitting
    function make_ts(; T = 120)
        ts = Date[]
        d = Date(2020, 1, 1)
        while length(ts) < T
            if dayofweek(d) <= 5
                push!(ts, d)
            end
            d += Day(1)
        end
        return ts
    end
    function make_prices(; T = 120, N = 5)
        rng = StableRNG(123456789)
        return TimeArray(make_ts(; T = T), 100 .+ cumsum(randn(rng, T, N) / 10; dims = 1),
                         string.("A", 1:N))
    end

    @testset "price-level splits mirror returns-level splits" begin
        X = make_prices()
        pr = PricesResult(; X = X)
        rd = ReturnsResult(; nx = string.("A", 1:5), X = values(X), ts = timestamp(X))

        for cv in (KFold(; n = 5), KFold(; n = 4, purged_size = 2, embargo_size = 1),
                   IndexWalkForward(60, 20), IndexWalkForward(60, 20; purged_size = 3),
                   DateWalkForward(60, 20),
                   DateWalkForward(8, 2; period = Week(1), previous = true))
            res_pr = split(cv, pr)
            res_rd = split(cv, rd)
            @test res_pr.train_idx == res_rd.train_idx
            @test res_pr.test_idx == res_rd.test_idx
            @test PortfolioOptimisers.n_splits(cv, pr) ==
                  PortfolioOptimisers.n_splits(cv, rd)
        end

        # helpers agree with the underlying data
        @test PortfolioOptimisers.cv_nobs(pr) == 120
        @test PortfolioOptimisers.cv_timestamps(pr) == timestamp(X)
        @test PortfolioOptimisers.cv_nobs(rd) == 120
        @test PortfolioOptimisers.cv_timestamps(rd) === rd.ts

        # combinatorial split is level-agnostic (it partitions observations), so it runs on
        # prices too and mirrors the returns-level split — the price-level combinatorial *fit*
        # accepts the boundary-return approximation over the non-contiguous training rows
        ccv0 = CombinatorialCrossValidation(; n_folds = 4, n_test_folds = 2)
        @test split(ccv0, pr).train_idx == split(ccv0, rd).train_idx
        @test split(ccv0, pr).test_idx == split(ccv0, rd).test_idx
        # multiple-randomised draws over assets (rows stay contiguous), so it is admissible
        # at the price level and mirrors the returns-level split
        mr = MultipleRandomised(IndexWalkForward(60, 20); subset_size = 3, n_subsets = 2,
                                seed = 42)
        mr_pr = split(mr, pr)
        mr_rd = split(mr, rd)
        @test mr_pr isa PortfolioOptimisers.MultipleRandomisedResult
        @test mr_pr.train_idx == mr_rd.train_idx
        @test mr_pr.test_idx == mr_rd.test_idx
        @test mr_pr.asset_idx == mr_rd.asset_idx

        # a Pipeline cannot be sub-selected by asset view, so it cannot be wrapped
        # in a meta-optimiser (ADR 0028 future expansion)
        pipe = Pipeline(; steps = (EmpiricalPrior(), EqualWeighted()))
        @test_throws ArgumentError PortfolioOptimisers.port_opt_view(pipe, 1:2)
        @test_throws ArgumentError optimise(pipe, rd)
    end

    @testset "predict applies fitted prep to the test window" begin
        X = make_prices()
        vals = copy(values(X))
        vals[1:50, 2] .= NaN     # A2: 62.5% missing in the train window -> dropped
        vals[10, 3] = NaN        # A3: sparse missing -> imputed
        Xm = TimeArray(timestamp(X), vals, string.("A", 1:5))
        pr = PricesResult(; X = Xm)

        pipe = Pipeline(;
                        steps = ("filter" => MissingDataFilter(; col_thr = 0.5),
                                 "impute" => Imputer(), PricesToReturns(), EmpiricalPrior(),
                                 EqualWeighted()))
        train_idx, test_idx = 1:80, 81:120
        res = fit(pipe, PortfolioOptimisers.port_opt_view(pr, train_idx))

        # the train window decided the universe
        @test res["filter"].nx == [:A1, :A3, :A4, :A5]
        @test length(res.w) == 4

        pred = PortfolioOptimisers.predict(res, pr, test_idx)

        # manual replay of the fitted steps on the test window
        pv = PortfolioOptimisers.port_opt_view(pr, test_idx)
        pv = PortfolioOptimisers.apply_preprocessing(res["filter"], pv)
        pv = PortfolioOptimisers.apply_preprocessing(res["impute"], pv)
        rd_test = PortfolioOptimisers.apply_preprocessing(PricesToReturns(), pv)
        pred_manual = PortfolioOptimisers.predict(res.ctx.opt, rd_test)
        @test pred.rd.X == pred_manual.rd.X

        # the T -> T-1 contraction: k price rows produce k-1 return rows
        @test size(pred.rd.X, 1) == length(test_idx) - 1

        # the test window is subset to the *train* universe even when clean
        pr_clean = PricesResult(; X = X)
        pred_clean = PortfolioOptimisers.predict(res, pr_clean, test_idx)
        @test size(pred_clean.rd.X, 2) == 1  # net portfolio returns column
        rd_clean = PortfolioOptimisers.apply_fitted_steps(res.results,
                                                          PortfolioOptimisers.port_opt_view(pr_clean,
                                                                                            test_idx))
        @test rd_clean.nx == ["A1", "A3", "A4", "A5"]

        # timestamp windows work too
        window_ts = timestamp(Xm)[test_idx]
        pred_ts = PortfolioOptimisers.predict(res, pr, window_ts)
        @test pred_ts.rd.X == pred.rd.X
    end

    @testset "predict at returns level" begin
        rng = StableRNG(987654321)
        rd = ReturnsResult(; nx = string.("A", 1:5), X = randn(rng, 100, 5) / 100)
        pipe = Pipeline(; steps = (EmpiricalPrior(), EqualWeighted()))
        res = fit(pipe, PortfolioOptimisers.port_opt_view(rd, 1:60, :))

        pred = PortfolioOptimisers.predict(res, rd, 61:100)
        pred_ref = PortfolioOptimisers.predict(res.ctx.opt, rd, collect(61:100))
        @test pred.rd.X ≈ pred_ref.rd.X

        # whole-data prediction with the default window
        pred_all = PortfolioOptimisers.predict(res, rd)
        @test size(pred_all.rd.X, 1) == 100
    end

    @testset "predict guards" begin
        X = make_prices(; T = 30)
        pr = PricesResult(; X = X)

        # no optimisation step -> no weights to predict with
        pipe = Pipeline(; steps = (PricesToReturns(), EmpiricalPrior()))
        res = fit(pipe, pr)
        @test_throws PortfolioOptimisers.IsNothingError PortfolioOptimisers.predict(res, pr,
                                                                                    1:10)

        # prices-level prediction requires a returns conversion among the fitted steps
        full = fit(Pipeline(; steps = (PricesToReturns(), EqualWeighted())), pr)
        broken = PortfolioOptimisers.PipelineResult(("filter",),
                                                    (PortfolioOptimisers.fit_preprocessing(MissingDataFilter(),
                                                                                           pr),),
                                                    full.ctx)
        @test_throws ArgumentError PortfolioOptimisers.predict(broken, pr, 1:10)
    end

    @testset "universe drift between train and test is an error" begin
        # PricesToReturns is stateless, and prices_to_returns drops assets that are
        # entirely missing in the window it converts. A train window in which one
        # asset is fully missing therefore yields fewer assets than a clean test
        # window -- weights and test returns would silently misalign.
        X = make_prices(; T = 60, N = 4)
        vals = copy(values(X))
        vals[1:30, 2] .= NaN     # A2 fully missing across the train window 1:30
        Xm = TimeArray(timestamp(X), vals, string.("A", 1:4))
        pr = PricesResult(; X = Xm)

        pipe = Pipeline(; steps = (PricesToReturns(), EmpiricalPrior(), EqualWeighted()))
        res = fit(pipe, PortfolioOptimisers.port_opt_view(pr, 1:30))
        @test res.ctx.returns.nx == ["A1", "A3", "A4"]
        @test length(res.w) == 3

        # predicting on a window where A2 is present must fail loudly, not misalign
        @test_throws ArgumentError PortfolioOptimisers.predict(res, pr, 31:60)

        # pinning the universe with a filter (and filling gaps) makes it well defined
        pipe_ok = Pipeline(;
                           steps = (MissingDataFilter(; col_thr = 0.5), Imputer(),
                                    PricesToReturns(), EmpiricalPrior(), EqualWeighted()))
        res_ok = fit(pipe_ok, PortfolioOptimisers.port_opt_view(pr, 1:30))
        @test res_ok.ctx.returns.nx == ["A1", "A3", "A4"]
        pred = PortfolioOptimisers.predict(res_ok, pr, 31:60)
        @test size(pred.rd.X, 1) == 29
    end

    @testset "nested pipelines replay recursively" begin
        X = make_prices()
        pr = PricesResult(; X = X)
        sub = Pipeline(; steps = (MissingDataFilter(), PricesToReturns()))
        pipe = Pipeline(; steps = (sub, EmpiricalPrior(), EqualWeighted()))
        res = fit(pipe, PortfolioOptimisers.port_opt_view(pr, 1:80))
        pred = PortfolioOptimisers.predict(res, pr, 81:120)
        @test size(pred.rd.X, 1) == 39
    end

    @testset "cross_val_predict over a pipeline" begin
        X = make_prices()
        pr = PricesResult(; X = X)
        cvw = IndexWalkForward(60, 30)
        sp = split(cvw, pr)
        nf = length(sp.train_idx)
        @test nf == 2
        ew, iv = EqualWeighted(), InverseVolatility()
        prep = (MissingDataFilter(), Imputer(), PricesToReturns(), EmpiricalPrior())
        static_pipe(opt) = Pipeline(; steps = (prep..., opt))
        function manual(opt, i)
            res = fit(static_pipe(opt),
                      PortfolioOptimisers.port_opt_view(pr, sp.train_idx[i]))
            return PortfolioOptimisers.predict(res, pr, sp.test_idx[i])
        end
        rng = StableRNG(192837465)
        rd_flat = ReturnsResult(; nx = string.("A", 1:5), X = randn(rng, 120, 5) / 100,
                                ts = make_ts())

        @testset "a static pipeline fits and predicts per fold" begin
            pipe = static_pipe(ew)
            @test !PortfolioOptimisers.is_time_dependent(pipe)
            @test !PortfolioOptimisers.needs_previous_weights(pipe)
            p = cross_val_predict(pipe, pr, cvw)
            @test length(p.pred) == nf
            for i in 1:nf
                m = manual(ew, i)
                @test isapprox(p.pred[i].res.w, m.res.w)
                @test isapprox(p.pred[i].rd.X, m.rd.X)
            end
            # returns-level input splits at its own level
            p2 = cross_val_predict(Pipeline(; steps = (EmpiricalPrior(), ew)), rd_flat,
                                   KFold(; n = 4))
            @test length(p2.pred) == 4
        end

        @testset "vector, functor and wrapped-callable schedules agree" begin
            sched = TimeDependent([ew, iv])
            pipe = Pipeline(; steps = (prep..., sched))
            # the statically-typed schedule forms classify as optimisation steps
            @test pipe.names[end] == "opt"
            @test PortfolioOptimisers.pipe_writes(sched) == :opt
            @test PortfolioOptimisers.pipe_reads(sched) == (:returns,)
            @test PortfolioOptimisers.is_time_dependent(pipe)
            p = cross_val_predict(pipe, pr, cvw)
            @test length(p.pred) == nf
            m1, m2 = manual(ew, 1), manual(iv, 2)
            @test isapprox(p.pred[1].res.w, m1.res.w)
            @test isapprox(p.pred[2].res.w, m2.res.w)
            @test !isapprox(p.pred[1].res.w, p.pred[2].res.w)
            # a declared functor schedules the same way, and its context carries the
            # fold's raw, pre-preprocessing input data
            cap = _test_TDPipeOpt(Vector{Any}(nothing, nf))
            pc = cross_val_predict(Pipeline(; steps = (prep..., TimeDependent(cap))), pr,
                                   cvw)
            @test isapprox(pc.pred[1].res.w, m1.res.w)
            @test isapprox(pc.pred[2].res.w, m2.res.w)
            @test all(x -> x === pr, cap.seen)
            # a bare callable enters via PipelineStep(writes = :opt)
            ps = PipelineStep(; est = TimeDependent(ctx -> isodd(ctx.i) ? ew : iv),
                              writes = :opt)
            pw = cross_val_predict(Pipeline(; steps = (prep..., ps)), pr, cvw)
            @test isapprox(pw.pred[1].res.w, m1.res.w)
            @test isapprox(pw.pred[2].res.w, m2.res.w)
            # its output is type-checked at the swap
            bad = PipelineStep(; est = TimeDependent(ctx -> EmpiricalPrior()),
                               writes = :opt)
            @test_throws ArgumentError cross_val_predict(Pipeline(; steps = (prep..., bad)),
                                                         pr, cvw,
                                                         ex = FLoops.SequentialEx())
        end

        @testset "mixed schedules: result folds predict, estimator folds optimise" begin
            r_pre = fit(static_pipe(iv), pr).ctx.opt
            p = cross_val_predict(Pipeline(; steps = (prep..., TimeDependent([ew, r_pre]))),
                                  pr, cvw)
            @test isapprox(p.pred[1].res.w, manual(ew, 1).res.w)
            # fold 2 replays the precomputed full-sample weights, not a fold-2 refit
            @test isapprox(p.pred[2].res.w, r_pre.w)
            @test !isapprox(r_pre.w, manual(iv, 2).res.w)
        end

        @testset "the fold's computed slots reach the fold's optimiser" begin
            hrp = HierarchicalRiskParity()
            capped = (MissingDataFilter(), Imputer(), PricesToReturns(),
                      WeightBoundsEstimator(; ub = 0.3))
            p = cross_val_predict(Pipeline(;
                                           steps = (capped..., TimeDependent([hrp, hrp]))),
                                  pr, cvw)
            @test all(pred -> all(w -> w <= 0.3 + 1e-8, pred.res.w), p.pred)
            # identical to the static pipeline: the injected slots reach the swapped-in
            # optimiser exactly as they reach a static step
            mp = cross_val_predict(Pipeline(; steps = (capped..., hrp)), pr, cvw)
            for i in 1:nf
                @test isapprox(p.pred[i].res.w, mp.pred[i].res.w)
            end
            # a result fold cannot consume computed constraints: fail closed
            r_pre = fit(static_pipe(iv), pr).ctx.opt
            bad = Pipeline(; steps = (capped..., TimeDependent([hrp, r_pre])))
            @test_throws ArgumentError cross_val_predict(bad, pr, cvw,
                                                         ex = FLoops.SequentialEx())
            # but a computed prior passes a result fold by
            okp = cross_val_predict(Pipeline(;
                                             steps = (prep..., TimeDependent([hrp, r_pre]))),
                                    pr, cvw)
            @test isapprox(okp.pred[2].res.w, r_pre.w)
        end

        @testset "previous weights thread through the fold loop" begin
            cap = _test_TDPipePrevW(Vector{Any}(nothing, nf))
            pipe = Pipeline(; steps = (prep..., TimeDependent(cap)))
            @test PortfolioOptimisers.needs_previous_weights(pipe)
            p = @test_logs (:info,) match_mode = :any cross_val_predict(pipe, pr, cvw)
            @test isnothing(cap.seen[1])
            @test cap.seen[2] ≈ p.pred[1].res.w
        end

        @testset "fold-less fit is default-or-throw" begin
            @test_throws PortfolioOptimisers.TimeDependentDefaultError fit(Pipeline(;
                                                                                    steps = (prep...,
                                                                                             TimeDependent([ew,
                                                                                                            iv]))),
                                                                           pr)
            pd = Pipeline(; steps = (prep..., TimeDependent([ew, iv]; default = iv)))
            res = fit(pd, pr)
            @test isapprox(res.w, fit(static_pipe(iv), pr).w)
        end

        @testset "construction and scheme guards" begin
            # a mis-sized schedule fails at the split, entries included
            @test_throws DimensionMismatch cross_val_predict(Pipeline(;
                                                                      steps = (prep...,
                                                                               TimeDependent([ew,
                                                                                              iv,
                                                                                              ew]))),
                                                             pr, cvw)
            # schedules of non-optimiser families are not steppable
            @test_throws ArgumentError PipelineStep(;
                                                    est = TimeDependent([EmpiricalPrior(),
                                                                         EmpiricalPrior()]),
                                                    writes = :prior)
            @test_throws ArgumentError PipelineStep(;
                                                    est = TimeDependent([EmpiricalPrior(),
                                                                         EmpiricalPrior()]),
                                                    writes = :opt)
            # an optimiser schedule step must write :opt
            @test_throws ArgumentError PipelineStep(; est = TimeDependent([ew, iv]),
                                                    writes = :prior)
            # a bare-callable schedule cannot infer its slot: wrap it
            @test_throws ArgumentError Pipeline(;
                                                steps = (prep..., TimeDependent(ctx -> ew)))
            # combinatorial and asset-resampling schemes now run for a price-starting pipeline
            # too (`static_pipe` begins with PricesToReturns): combinatorial over non-contiguous
            # training rows (boundary-return approximation), MR over asset subsets. Depth is in
            # their own testsets; here just confirm they run.
            @test isa(cross_val_predict(static_pipe(ew), pr,
                                        CombinatorialCrossValidation(; n_folds = 4,
                                                                     n_test_folds = 2)),
                      PortfolioOptimisers.PopulationPredictionResult)
            @test isa(cross_val_predict(static_pipe(ew), pr,
                                        MultipleRandomised(IndexWalkForward(60, 30);
                                                           subset_size = 3, n_subsets = 2,
                                                           seed = 1)),
                      PortfolioOptimisers.PopulationPredictionResult)
            # one evaluation protocol per call: a holdout pipeline is rejected
            @test_throws ArgumentError cross_val_predict(Pipeline(;
                                                                  steps = (TrainTestSplit(;
                                                                                          test_size = 0.2),
                                                                           EmpiricalPrior(),
                                                                           ew)), rd_flat,
                                                         cvw)
        end

        @testset "tuning folds resolve schedules in search CV" begin
            cvt = IndexWalkForward(60, 30)
            spt = split(cvt, rd_flat)
            M = length(spt.train_idx)
            r = ConditionalValueatRisk()
            pipe = Pipeline(; steps = (EmpiricalPrior(), ew))
            # constant schedules score identically to their static candidates
            res = search_cross_validation(pipe,
                                          GridSearchCrossValidation(["opt" =>
                                                                         [TimeDependent(fill(ew,
                                                                                             M)),
                                                                          TimeDependent(fill(iv,
                                                                                             M))]];
                                                                    cv = cvt, r = r),
                                          rd_flat)
            rref = search_cross_validation(pipe,
                                           GridSearchCrossValidation(["opt" => [ew, iv]];
                                                                     cv = cvt, r = r),
                                           rd_flat)
            @test size(res.test_scores) == (M, 2)
            @test res.test_scores ≈ rref.test_scores
            # tuning fold j runs entry j
            rmix = search_cross_validation(pipe,
                                           GridSearchCrossValidation(["opt" =>
                                                                          [TimeDependent([ew,
                                                                                          iv])]];
                                                                     cv = cvt, r = r),
                                           rd_flat)
            @test rmix.test_scores[1, 1] ≈ rref.test_scores[1, 1]
            @test rmix.test_scores[2, 1] ≈ rref.test_scores[2, 2]
            # a candidate's schedule must match the tuning fold count
            @test_throws DimensionMismatch search_cross_validation(pipe,
                                                                   GridSearchCrossValidation(["opt" =>
                                                                                                  [TimeDependent([ew,
                                                                                                                  iv,
                                                                                                                  ew])]];
                                                                                             cv = cvt,
                                                                                             r = r,
                                                                                             ex = FLoops.SequentialEx()),
                                                                   rd_flat)
        end

        @testset "combinatorial and MultipleRandomised over a returns-level pipeline" begin
            # A returns-level pipeline runs both exactly (no rolling transform), like the
            # plain-optimiser loops. `rpipe` has no rolling/price step. (Price-starting
            # pipelines also run them — combinatorial with a boundary-return approximation —
            # covered in the price-level testsets.)
            rpipe(opt) = Pipeline(; steps = (EmpiricalPrior(), opt))
            @testset "combinatorial" begin
                ccv = CombinatorialCrossValidation(; n_folds = 4, n_test_folds = 2)
                n_c = PortfolioOptimisers.n_splits(ccv)
                pp = cross_val_predict(rpipe(iv), rd_flat, ccv)
                @test isa(pp, PortfolioOptimisers.PopulationPredictionResult)
                @test all(isa(p.res.retcode, PortfolioOptimisers.OptimisationSuccess)
                          for pa in pp.pred for p in pa.pred)
                # A homogeneous schedule reproduces the pure run at every position (the
                # schedule is consumed per split); a mixed schedule runs and switches.
                homog = cross_val_predict(rpipe(TimeDependent(fill(iv, n_c))), rd_flat, ccv)
                @test [length(p.pred) for p in homog.pred] ==
                      [length(p.pred) for p in pp.pred]
                @test all(isapprox(a.res.w, b.res.w)
                          for (pa, pb) in zip(homog.pred, pp.pred)
                          for (a, b) in zip(pa.pred, pb.pred))
                mixed = cross_val_predict(rpipe(TimeDependent([iseven(j) ? iv : ew
                                                               for j in 1:n_c])), rd_flat,
                                          ccv)
                @test all(isa(p.res.retcode, PortfolioOptimisers.OptimisationSuccess)
                          for pa in mixed.pred for p in pa.pred)
            end
            @testset "MultipleRandomised" begin
                inner = IndexWalkForward(60, 30)
                n_pp = PortfolioOptimisers.n_splits(inner, rd_flat)
                mkmr = () -> MultipleRandomised(inner; subset_size = 3, n_subsets = 2,
                                                rng = StableRNG(42), seed = 1)
                pm = cross_val_predict(rpipe(iv), rd_flat, mkmr())
                @test isa(pm, PortfolioOptimisers.PopulationPredictionResult)
                @test length(pm.pred) == 2                       # n_subsets paths
                @test all(path -> length(path.pred) == n_pp, pm.pred)
                @test all(length(p.res.w) == 3 for pa in pm.pred for p in pa.pred)
                @test all(isa(p.res.retcode, PortfolioOptimisers.OptimisationSuccess)
                          for pa in pm.pred for p in pa.pred)
                # Homogeneous schedule reproduces the pure run (same seed => same split).
                homog = cross_val_predict(rpipe(TimeDependent(fill(iv, n_pp))), rd_flat,
                                          mkmr())
                @test all(isapprox(a.res.w, b.res.w)
                          for (pa, pb) in zip(homog.pred, pm.pred)
                          for (a, b) in zip(pa.pred, pb.pred))
                # A wrong-length schedule is rejected per path.
                @test_throws DimensionMismatch cross_val_predict(rpipe(TimeDependent([iv])),
                                                                 rd_flat, mkmr())
            end
        end

        @testset "MultipleRandomised over a price-starting pipeline" begin
            # MR resamples assets; every observation window stays contiguous, so — unlike
            # combinatorial — a price-starting pipeline (PricesToReturns) is admissible.
            inner = IndexWalkForward(60, 30)
            n_pp = PortfolioOptimisers.n_splits(inner, pr)
            mr = MultipleRandomised(inner; subset_size = 3, n_subsets = 2,
                                    rng = StableRNG(42), seed = 1)

            pm = cross_val_predict(static_pipe(iv), pr, mr)
            @test isa(pm, PortfolioOptimisers.PopulationPredictionResult)
            @test length(pm.pred) == 2                       # n_subsets paths
            @test all(path -> length(path.pred) == n_pp, pm.pred)
            @test all(length(p.res.w) == 3 for pa in pm.pred for p in pa.pred)
            @test all(isa(p.res.retcode, PortfolioOptimisers.OptimisationSuccess)
                      for pa in pm.pred for p in pa.pred)

            # combinatorial also runs at the price level (boundary-return approximation over
            # its non-contiguous training rows); test groups are contiguous so predictions hold
            ccv = CombinatorialCrossValidation(; n_folds = 4, n_test_folds = 2)
            cc = cross_val_predict(static_pipe(iv), pr, ccv)
            @test isa(cc, PortfolioOptimisers.PopulationPredictionResult)
            @test length(cc.pred) == maximum(split(ccv, pr).path_ids)
            @test all(isa(p.res.retcode, PortfolioOptimisers.OptimisationSuccess)
                      for pa in cc.pred for p in pa.pred)
        end
    end
end
