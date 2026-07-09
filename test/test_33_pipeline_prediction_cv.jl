@testset "Pipeline prediction and CV" begin
    using Test, PortfolioOptimisers, TimeSeries, Dates, StableRNGs, Statistics

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

        # returns-level-only CV schemes fail loudly on price-level data
        @test_throws ArgumentError split(CombinatorialCrossValidation(), pr)
        @test_throws ArgumentError split(MultipleRandomised(IndexWalkForward(60, 20)), pr)

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
end
