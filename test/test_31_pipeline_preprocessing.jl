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
        @test PortfolioOptimisers.fit_preprocessing(ptr, pr) === ptr

        # apply matches the underlying function
        rr = PortfolioOptimisers.apply_preprocessing(ptr, pr)
        rr_ref = prices_to_returns(X)
        @test rr.X == rr_ref.X
        @test rr.nx == rr_ref.nx
        @test rr.ts == rr_ref.ts

        # ret_method is honoured
        rr_log = PortfolioOptimisers.apply_preprocessing(PricesToReturns(;
                                                                         ret_method = :log),
                                                         pr)
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
        # The domain is `[0, 1]`: zero is the tightest policy the estimator can state, and
        # it is a spelling rather than an arithmetic accident (ADR 0133).
        @test_throws DomainError MissingDataFilter(; col_thr = -0.1)
        @test_throws DomainError MissingDataFilter(; col_thr = 1.5)
        @test_throws DomainError MissingDataFilter(; row_thr = -0.1)
        @test_throws DomainError MissingDataFilter(; row_thr = 1.5)
        @test MissingDataFilter(; col_thr = 0.0, row_thr = 0.0) isa MissingDataFilter

        X, ts = make_prices()
        vals = copy(values(X))
        vals[1:6, 2] .= NaN      # A2: 60% missing -> dropped at col_thr = 0.5
        vals[1, 3] = NaN         # A3: 10% missing -> kept
        Xm = TimeArray(ts, vals, string.("A", 1:4))
        pr = PricesResult(; X = Xm)

        mdf = MissingDataFilter(; col_thr = 0.5, row_thr = 0.5)
        res = PortfolioOptimisers.fit_preprocessing(mdf, pr)
        @test res.nx == [:A1, :A3, :A4]

        # apply subsets the universe; row 1 has 1/3 missing <= row_thr so it stays
        pv = PortfolioOptimisers.apply_preprocessing(res, pr)
        @test TimeSeries.colnames(pv.X) == [:A1, :A3, :A4]
        @test length(TimeSeries.timestamp(pv.X)) == 10

        # rows above the row threshold are dropped from the window being transformed
        res_strict = PortfolioOptimisers.fit_preprocessing(MissingDataFilter(;
                                                                             col_thr = 0.5,
                                                                             row_thr = 0.2),
                                                           pr)
        pv_strict = PortfolioOptimisers.apply_preprocessing(res_strict, pr)
        @test length(TimeSeries.timestamp(pv_strict.X)) == 9
        @test first(TimeSeries.timestamp(pv_strict.X)) == ts[2]

        # the universe is fitted state: a clean test window is still subset to it
        Xc, _ = make_prices()
        pr_clean = PricesResult(; X = Xc)
        pv_clean = PortfolioOptimisers.apply_preprocessing(res, pr_clean)
        @test TimeSeries.colnames(pv_clean.X) == [:A1, :A3, :A4]

        # iv and vector ivpa follow the universe
        iv = TimeArray(ts, rand(StableRNG(4), 10, 4), string.("A", 1:4))
        pr_iv = PricesResult(; X = Xm, iv = iv, ivpa = [0.1, 0.2, 0.3, 0.4])
        pv_iv = PortfolioOptimisers.apply_preprocessing(res, pr_iv)
        @test TimeSeries.colnames(pv_iv.iv) == [:A1, :A3, :A4]
        @test pv_iv.ivpa == [0.1, 0.3, 0.4]

        # dropping every asset is an error at fit time
        Xall = TimeArray(ts, fill(NaN, 10, 2), ["A1", "A2"])
        @test_throws PortfolioOptimisers.IsEmptyError PortfolioOptimisers.fit_preprocessing(MissingDataFilter(;
                                                                                                              col_thr = 0.5),
                                                                                            PricesResult(;
                                                                                                         X = Xall))

        # run_step reads and writes :prices
        ctx = PortfolioOptimisers.PipelineContext(; prices = pr)
        fitted, ctx2 = PortfolioOptimisers.run_step(mdf, ctx)
        @test fitted isa MissingDataFilterResult
        @test TimeSeries.colnames(ctx2.prices.X) == [:A1, :A3, :A4]
    end

    @testset "MissingDataFilter at the zero threshold" begin
        # Map #955, ADR 0133. `prices_to_returns` deletes nothing, so this estimator is the
        # only filter there is, and `0.0` is the tightest policy it can state: no gap is
        # tolerated. It is what `dropmissing!` used to do inside the conversion, said by
        # name and split correctly across the fit/apply seam.
        ts = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 6))
        # `A1` is priced throughout; `A2` and `A5` are listed from observation 3; `A3` is
        # suspended at observation 4; `A4` is delisted after observation 4. The two
        # inceptions make the leading rows the gappiest, so a row threshold between them
        # and the rest opens the window without touching the delisting.
        vals = [10.0 NaN 30.0 40.0 NaN
                11.0 NaN 31.0 41.0 NaN
                12.0 20.0 32.0 42.0 50.0
                13.0 21.0 NaN 43.0 51.0
                14.0 22.0 34.0 NaN 52.0
                15.0 23.0 35.0 NaN 53.0]
        Xg = TimeArray(ts, vals, string.("A", 1:5))
        pr = PricesResult(; X = Xg, span = listing_span(vals))

        # `col_thr = 0.0` keeps the columns with no gap at all, and nothing else. Only `A1`
        # is priced at every observation of the training window.
        zc = PortfolioOptimisers.fit_preprocessing(MissingDataFilter(; col_thr = 0.0), pr)
        @test zc.nx == [:A1]
        # A column holding one gap in six is above zero, so the boundary is `count == 0`
        # and not a small fraction: `A3` needs the threshold raised to its own fraction.
        @test PortfolioOptimisers.fit_preprocessing(MissingDataFilter(; col_thr = 1 / 6),
                                                    pr).nx == [:A1, :A3]

        # `row_thr = 0.0` keeps the observations at which every surviving asset is priced.
        # Over the whole universe that is observation 3 alone.
        zr = PortfolioOptimisers.fit_preprocessing(MissingDataFilter(; col_thr = 1.0,
                                                                     row_thr = 0.0), pr)
        pv = PortfolioOptimisers.apply_preprocessing(zr, pr)
        @test TimeSeries.timestamp(pv.X) == [ts[3]]
        @test !any(PortfolioOptimisers.is_missing_value, values(pv.X))
        # A row holding one gap in five is above zero, so the row boundary is `count == 0`
        # too, and it takes `1/5` to admit one.
        loose = PortfolioOptimisers.fit_preprocessing(MissingDataFilter(; col_thr = 1.0,
                                                                        row_thr = 1 / 5),
                                                      pr)
        @test TimeSeries.timestamp(PortfolioOptimisers.apply_preprocessing(loose, pr).X) ==
              ts[[3, 4, 5, 6]]

        # A **column** drop leaves every row where it was, so the Span Rule reads the same
        # `first` and `last` off the filtered panel as off the raw one for every asset the
        # filter kept. This is the reading the ticket asked for, and it holds on the axis
        # the fitted universe cuts.
        raw = listing_span(vals)
        conly = PortfolioOptimisers.fit_preprocessing(MissingDataFilter(; col_thr = 1 / 6,
                                                                        row_thr = 1.0), pr)
        cpr = PortfolioOptimisers.apply_preprocessing(conly, pr)
        @test conly.nx == [:A1, :A3]
        cut = listing_span(values(cpr.X))
        cols = Vector{Int}(indexin(conly.nx, TimeSeries.colnames(pr.X)))
        @test cut.first == raw.first[cols]
        @test cut.last == raw.last[cols]

        # A **row** drop does move an observation, so re-deriving the rule off the filtered
        # panel is *not* the same reading — which is exactly why the estimator never
        # re-derives. The span it hands on is the carrier's, viewed at the surviving rows
        # and columns, so every surviving observation says of every kept asset what it said
        # before the filter ran.
        for thr in (0.0, 1 / 5, 2 / 5, 1.0)
            res = PortfolioOptimisers.fit_preprocessing(MissingDataFilter(; col_thr = 1.0,
                                                                          row_thr = thr),
                                                        pr)
            fpr = PortfolioOptimisers.apply_preprocessing(res, pr)
            rows = Vector{Int}(indexin(TimeSeries.timestamp(fpr.X), ts))
            @test fpr.span == view(pr.span, rows, 1:length(res.nx))
        end

        # And that is what keeps a delisting readable. At `row_thr = 1/5` the window opens
        # at observation 3, and `A4`'s two trailing gaps are still outside its span.
        f4 = PortfolioOptimisers.apply_preprocessing(loose, pr)
        @test TimeSeries.timestamp(f4.X) == ts[[3, 4, 5, 6]]
        @test f4.span[:, 4] == [true, true, false, false]
        # The inception `A2` and `A5` carry is inside the surviving window on both
        # readings, so nothing about them turns on the view here.
        @test f4.span[:, 2] == [true, true, true, true]

        # The case that proves a re-derivation would be wrong rather than merely different:
        # a row drop can take away every priced observation an asset has inside its
        # listing. `B2` is priced at observations 2 and 5 and suspended between them, and
        # `row_thr = 0.0` keeps observations 3 and 4 alone. The carried span still reports
        # it listed there -- a Held Gap -- where the rule read off the surviving prices
        # would report an asset that was never listed at all.
        ts2 = collect(Date(2020, 1, 1):Day(1):Date(2020, 1, 6))
        v2 = [10.0 NaN
              11.0 20.0
              12.0 NaN
              13.0 NaN
              14.0 23.0
              NaN NaN]
        pr2 = PricesResult(; X = TimeArray(ts2, v2, ["B1", "B2"]), span = listing_span(v2))
        # Observations 3 and 4 hold a gap in `B2` alone, so no row survives `row_thr = 0`;
        # `1/2` keeps every row whose gap count is at most one, which is 1 through 5.
        r2 = PortfolioOptimisers.fit_preprocessing(MissingDataFilter(; col_thr = 1.0,
                                                                     row_thr = 1 / 2), pr2)
        f2 = PortfolioOptimisers.apply_preprocessing(r2, pr2)
        @test TimeSeries.timestamp(f2.X) == ts2[1:5]
        # The carried span says B2 is listed from observation 2 to observation 5.
        @test f2.span[:, 2] == [false, true, true, true, true]
        # A re-derivation would agree here, because the priced observations survived. Now
        # take them away: keep observations 3 and 4 by hand and compare the two readings.
        kept_rows = [3, 4]
        @test view(pr2.span, kept_rows, :)[:, 2] == [true, true]
        rederived = listing_span(v2[kept_rows, :])
        @test rederived.last[2] < rederived.first[2]
    end

    @testset "Imputer" begin
        X, ts = make_prices()
        vals = copy(values(X))
        vals[2, 1] = NaN
        Xm = TimeArray(ts, vals, string.("A", 1:4))
        pr = PricesResult(; X = Xm)

        imp = Imputer()
        res = PortfolioOptimisers.fit_preprocessing(imp, pr)
        @test res.nx == [:A1, :A2, :A3, :A4]
        train_med = median([x for x in vals[:, 1] if !isnan(x)])
        @test res.v[1] == train_med

        pv = PortfolioOptimisers.apply_preprocessing(res, pr)
        @test values(pv.X)[2, 1] == train_med
        @test !any(isnan, values(pv.X))

        # leakage regression: a test window is filled with TRAIN statistics,
        # not with its own
        Xt, _ = make_prices()
        tvals = copy(values(Xt)) .+ 50.0
        tvals[5, 1] = NaN
        Xtm = TimeArray(ts, tvals, string.("A", 1:4))
        pv_test = PortfolioOptimisers.apply_preprocessing(res, PricesResult(; X = Xtm))
        test_med = median([x for x in tvals[:, 1] if !isnan(x)])
        @test values(pv_test.X)[5, 1] == train_med
        @test values(pv_test.X)[5, 1] != test_med

        # the statistic is configurable
        res_mean = PortfolioOptimisers.fit_preprocessing(Imputer(; stat = MeanValue()), pr)
        train_mean = mean([x for x in vals[:, 1] if !isnan(x)])
        @test res_mean.v[1] ≈ train_mean

        # assets with no observed values get no fill value and pass through
        vals_empty = copy(values(X))
        vals_empty[:, 2] .= NaN
        Xe = TimeArray(ts, vals_empty, string.("A", 1:4))
        res_e = PortfolioOptimisers.fit_preprocessing(imp, PricesResult(; X = Xe))
        @test res_e.nx == [:A1, :A3, :A4]
        pv_e = PortfolioOptimisers.apply_preprocessing(res_e, PricesResult(; X = Xe))
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
