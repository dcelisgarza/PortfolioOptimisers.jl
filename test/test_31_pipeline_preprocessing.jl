@testset "Pipeline preprocessing steps" begin
    using Test, PortfolioOptimisers, TimeSeries, Dates, StableRNGs, Statistics

    function make_prices(; T = 10, N = 4)
        rng = StableRNG(123456789)
        ts = Date(2020, 1, 1):Day(1):Date(2020, 1, T)
        X = TimeArray(ts, 100 .+ rand(rng, T, N), string.("A", 1:N))
        return X, ts
    end

    @testset "PricesToReturns" begin
        @test_throws ArgumentError PricesToReturns(; ret_method = :nonsense)

        X, _ = make_prices()
        pr = PricesResult(; X = X)
        ptr = PricesToReturns()

        # stateless: the fitted object is the estimator itself
        @test PortfolioOptimisers.fit_step(ptr, pr) === ptr

        # apply matches the underlying function
        rr = PortfolioOptimisers.apply_step(ptr, pr)
        rr_ref = prices_to_returns(X)
        @test rr.X == rr_ref.X
        @test rr.nx == rr_ref.nx
        @test rr.ts == rr_ref.ts

        # ret_method is honoured
        rr_log = PortfolioOptimisers.apply_step(PricesToReturns(; ret_method = :log), pr)
        @test rr_log.X != rr.X
        @test rr_log.X ≈ log1p.(rr.X)

        # run_step reads :prices, writes :returns
        ctx = PortfolioOptimisers.PipelineContext(; prices = pr)
        fitted, ctx2 = PortfolioOptimisers.run_step(ptr, ctx)
        @test fitted === ptr
        @test ctx2.returns.X == rr.X
        @test ctx2.prices === pr

        # requires :prices
        @test_throws PortfolioOptimisers.IsNothingError PortfolioOptimisers.run_step(ptr,
                                                                                     PortfolioOptimisers.PipelineContext())
    end

    @testset "MissingDataFilter" begin
        @test_throws DomainError MissingDataFilter(; col_thr = 0.0)
        @test_throws DomainError MissingDataFilter(; col_thr = 1.5)
        @test_throws DomainError MissingDataFilter(; row_thr = 0.0)

        X, ts = make_prices()
        vals = copy(values(X))
        vals[1:6, 2] .= NaN      # A2: 60% missing -> dropped at col_thr = 0.5
        vals[1, 3] = NaN         # A3: 10% missing -> kept
        Xm = TimeArray(ts, vals, string.("A", 1:4))
        pr = PricesResult(; X = Xm)

        mdf = MissingDataFilter(; col_thr = 0.5, row_thr = 0.5)
        res = PortfolioOptimisers.fit_step(mdf, pr)
        @test res.nx == [:A1, :A3, :A4]

        # apply subsets the universe; row 1 has 1/3 missing <= row_thr so it stays
        pv = PortfolioOptimisers.apply_step(res, pr)
        @test TimeSeries.colnames(pv.X) == [:A1, :A3, :A4]
        @test length(TimeSeries.timestamp(pv.X)) == 10

        # rows above the row threshold are dropped from the window being transformed
        res_strict = PortfolioOptimisers.fit_step(MissingDataFilter(; col_thr = 0.5,
                                                                    row_thr = 0.2), pr)
        pv_strict = PortfolioOptimisers.apply_step(res_strict, pr)
        @test length(TimeSeries.timestamp(pv_strict.X)) == 9
        @test first(TimeSeries.timestamp(pv_strict.X)) == ts[2]

        # the universe is fitted state: a clean test window is still subset to it
        Xc, _ = make_prices()
        pr_clean = PricesResult(; X = Xc)
        pv_clean = PortfolioOptimisers.apply_step(res, pr_clean)
        @test TimeSeries.colnames(pv_clean.X) == [:A1, :A3, :A4]

        # iv and vector ivpa follow the universe
        iv = TimeArray(ts, rand(StableRNG(4), 10, 4), string.("A", 1:4))
        pr_iv = PricesResult(; X = Xm, iv = iv, ivpa = [0.1, 0.2, 0.3, 0.4])
        pv_iv = PortfolioOptimisers.apply_step(res, pr_iv)
        @test TimeSeries.colnames(pv_iv.iv) == [:A1, :A3, :A4]
        @test pv_iv.ivpa == [0.1, 0.3, 0.4]

        # dropping every asset is an error at fit time
        Xall = TimeArray(ts, fill(NaN, 10, 2), ["A1", "A2"])
        @test_throws PortfolioOptimisers.IsEmptyError PortfolioOptimisers.fit_step(MissingDataFilter(;
                                                                                                     col_thr = 0.5),
                                                                                   PricesResult(;
                                                                                                X = Xall))

        # run_step reads and writes :prices
        ctx = PortfolioOptimisers.PipelineContext(; prices = pr)
        fitted, ctx2 = PortfolioOptimisers.run_step(mdf, ctx)
        @test fitted isa MissingDataFilterResult
        @test TimeSeries.colnames(ctx2.prices.X) == [:A1, :A3, :A4]
    end

    @testset "Imputer" begin
        X, ts = make_prices()
        vals = copy(values(X))
        vals[2, 1] = NaN
        Xm = TimeArray(ts, vals, string.("A", 1:4))
        pr = PricesResult(; X = Xm)

        imp = Imputer()
        res = PortfolioOptimisers.fit_step(imp, pr)
        @test res.nx == [:A1, :A2, :A3, :A4]
        train_med = median([x for x in vals[:, 1] if !isnan(x)])
        @test res.v[1] == train_med

        pv = PortfolioOptimisers.apply_step(res, pr)
        @test values(pv.X)[2, 1] == train_med
        @test !any(isnan, values(pv.X))

        # leakage regression: a test window is filled with TRAIN statistics,
        # not with its own
        Xt, _ = make_prices()
        tvals = copy(values(Xt)) .+ 50.0
        tvals[5, 1] = NaN
        Xtm = TimeArray(ts, tvals, string.("A", 1:4))
        pv_test = PortfolioOptimisers.apply_step(res, PricesResult(; X = Xtm))
        test_med = median([x for x in tvals[:, 1] if !isnan(x)])
        @test values(pv_test.X)[5, 1] == train_med
        @test values(pv_test.X)[5, 1] != test_med

        # the statistic is configurable
        res_mean = PortfolioOptimisers.fit_step(Imputer(; stat = MeanValue()), pr)
        train_mean = mean([x for x in vals[:, 1] if !isnan(x)])
        @test res_mean.v[1] ≈ train_mean

        # assets with no observed values get no fill value and pass through
        vals_empty = copy(values(X))
        vals_empty[:, 2] .= NaN
        Xe = TimeArray(ts, vals_empty, string.("A", 1:4))
        res_e = PortfolioOptimisers.fit_step(imp, PricesResult(; X = Xe))
        @test res_e.nx == [:A1, :A3, :A4]
        pv_e = PortfolioOptimisers.apply_step(res_e, PricesResult(; X = Xe))
        @test all(isnan, values(pv_e.X)[:, 2])

        # run_step reads and writes :prices
        ctx = PortfolioOptimisers.PipelineContext(; prices = pr)
        fitted, ctx2 = PortfolioOptimisers.run_step(imp, ctx)
        @test fitted isa ImputerResult
        @test !any(isnan, values(ctx2.prices.X))
    end

    @testset "run_step for existing estimator families" begin
        rng = StableRNG(987654321)
        rd = ReturnsResult(; nx = string.("A", 1:5), X = randn(rng, 60, 5) / 100)
        ctx = PortfolioOptimisers.PipelineContext(; returns = rd)

        # prior step
        fitted, ctx2 = PortfolioOptimisers.run_step(EmpiricalPrior(), ctx)
        @test fitted === ctx2.prior
        @test ctx2.prior.X == rd.X
        @test length(ctx2.prior.mu) == 5

        # phylogeny step
        fitted, ctx3 = PortfolioOptimisers.run_step(ClustersEstimator(), ctx2)
        @test fitted === ctx3.phylogeny
        @test !isnothing(ctx3.phylogeny)
        @test ctx3.prior === ctx2.prior

        # optimisation step (naive, no solver required)
        fitted, ctx4 = PortfolioOptimisers.run_step(EqualWeighted(), ctx3)
        @test fitted === ctx4.opt
        @test fitted.w ≈ fill(0.2, 5)

        # steps require their slots
        @test_throws PortfolioOptimisers.IsNothingError PortfolioOptimisers.run_step(EmpiricalPrior(),
                                                                                     PortfolioOptimisers.PipelineContext())

        # non-steppable estimators are rejected with guidance
        @test_throws ArgumentError PortfolioOptimisers.run_step(Covariance(), ctx)
    end

    @testset "PipelineStep execution" begin
        rng = StableRNG(987654321)
        rd = ReturnsResult(; nx = string.("A", 1:5), X = randn(rng, 60, 5) / 100)
        ctx = PortfolioOptimisers.PipelineContext(; returns = rd)

        # wrapped callable: receives the context, its value lands in the declared slot
        ps = PipelineStep(; est = c -> prior(EmpiricalPrior(), c.returns),
                          reads = (:returns,), writes = :prior)
        fitted, ctx2 = PortfolioOptimisers.run_step(ps, ctx)
        @test fitted === ctx2.prior
        @test ctx2.prior.X == rd.X

        # declared reads are enforced
        @test_throws PortfolioOptimisers.IsNothingError PortfolioOptimisers.run_step(ps,
                                                                                     PortfolioOptimisers.PipelineContext())

        # wrapped estimators delegate to their family's run_step
        ps2 = PipelineStep(; est = EmpiricalPrior(), reads = (:returns,), writes = :prior)
        fitted2, ctx3 = PortfolioOptimisers.run_step(ps2, ctx)
        @test fitted2.mu ≈ fitted.mu
    end
end
