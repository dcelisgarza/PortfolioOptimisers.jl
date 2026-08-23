@testset "Train/test split" begin
    using Test, PortfolioOptimisers, TimeSeries, Dates, StableRNGs

    safe_index = PortfolioOptimisers.safe_index

    function make_prices(; T = 100, N = 4)
        rng = StableRNG(987654321)
        ts = Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(T - 1))
        X = TimeArray(collect(ts), 100 .+ cumsum(rand(rng, T, N); dims = 1),
                      string.("A", 1:N))
        return PricesResult(; X = X)
    end
    function make_returns(; T = 100, N = 4)
        rng = StableRNG(987654321)
        ts = collect(Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(T - 1)))
        return ReturnsResult(; nx = string.("A", 1:N), X = 0.01 .* randn(rng, T, N),
                             ts = ts)
    end

    @testset "safe_index sizing" begin
        # neither size given: 75/25, and the two windows partition the data
        @test safe_index(nothing, nothing, 100) == (1:75, 76:100)

        # one side given: the other is its complement, counts and fractions alike
        @test safe_index(80, nothing, 100) == (1:80, 81:100)
        @test safe_index(0.8, nothing, 100) == (1:80, 81:100)
        @test safe_index(nothing, 20, 100) == (1:80, 81:100)
        @test safe_index(nothing, 0.2, 100) == (1:80, 81:100)

        # both given and summing to less than N: the middle rows are embargoed
        @test safe_index(0.6, 0.2, 100) == (1:60, 81:100)
        @test safe_index(60, 20, 100) == (1:60, 81:100)

        # both given and exactly covering the data: no embargo
        @test safe_index(70, 30, 100) == (1:70, 71:100)

        # overlapping windows are rejected
        @test_throws ArgumentError safe_index(0.9, 0.2, 100)
        @test_throws ArgumentError safe_index(90, 20, 100)

        # a window that would be empty is rejected rather than silently returned
        @test_throws ArgumentError safe_index(100, nothing, 100)
        @test_throws ArgumentError safe_index(nothing, 100, 100)

        # counts saturate, fractions must be a genuine fraction
        @test_throws ArgumentError safe_index(150, nothing, 100)  # saturates, empties test
        @test_throws DomainError safe_index(0.0, nothing, 100)
        @test_throws DomainError safe_index(1.0, nothing, 100)
        @test_throws DomainError safe_index(nothing, 1.5, 100)
        @test_throws DomainError safe_index(0, nothing, 100)
        @test_throws DomainError safe_index(nothing, -5, 100)
    end

    @testset "split_count boundaries" begin
        split_count = PortfolioOptimisers.split_count

        # A count saturates at the observation total.
        @test split_count(3, 10, :train_size) == 3
        @test split_count(10, 10, :train_size) == 10
        @test split_count(15, 10, :train_size) == 10

        # A fraction that lands exactly on an integer takes that integer, and one that
        # floors below one row is clamped up to one, so rounding alone never empties a
        # window.
        @test split_count(0.5, 10, :train_size) == 5
        @test split_count(0.01, 10, :train_size) == 1
        @test split_count(0.9999, 10, :train_size) == 9

        # A count must be positive, and a fraction must lie strictly inside (0, 1).
        @test_throws DomainError split_count(0, 10, :train_size)
        @test_throws DomainError split_count(-1, 10, :train_size)
        @test_throws DomainError split_count(0.0, 10, :train_size)
        @test_throws DomainError split_count(1.0, 10, :train_size)
    end

    @testset "train_test_split free function" begin
        pr = make_prices()
        rd = make_returns()

        tr, te = train_test_split(rd; test_size = 0.2)
        @test size(tr.X, 1) == 80
        @test size(te.X, 1) == 20
        @test tr.X == rd.X[1:80, :]
        @test te.X == rd.X[81:100, :]
        @test tr.nx == rd.nx == te.nx

        trp, tep = train_test_split(pr; test_size = 0.2)
        @test size(values(trp.X), 1) == 80
        @test size(values(tep.X), 1) == 20
        @test timestamp(tep.X) == timestamp(pr.X)[81:100]

        # the previously-broken branches now yield a non-empty test window
        tr2, te2 = train_test_split(rd)
        @test size(tr2.X, 1) == 75
        @test size(te2.X, 1) == 25
        tr3, te3 = train_test_split(rd; train_size = 60)
        @test size(tr3.X, 1) == 60
        @test size(te3.X, 1) == 40

        # embargo: the rows between the windows belong to neither
        tr4, te4 = train_test_split(rd; train_size = 0.6, test_size = 0.2)
        @test size(tr4.X, 1) == 60
        @test size(te4.X, 1) == 20
        @test te4.X == rd.X[81:100, :]
    end

    @testset "TrainTestSplit estimator" begin
        pr = make_prices()
        rd = make_returns()

        tts = TrainTestSplit(; test_size = 0.2)
        @test TTS === TrainTestSplit

        # the estimator form of the free function returns the same fitted result
        tsr = train_test_split(tts, rd)
        @test isa(tsr, TrainTestSplitResult)
        @test size(tsr.train.X, 1) == 80
        @test size(tsr.test.X, 1) == 20
        @test tsr.test.X == rd.X[81:100, :]
        @test size(values(train_test_split(tts, pr).test.X), 1) == 20

        # fitting cuts both windows; applying is a pass-through
        res = PortfolioOptimisers.fit_preprocessing(tts, rd)
        @test isa(res, TrainTestSplitResult)
        @test size(res.train.X, 1) == 80
        @test size(res.test.X, 1) == 20
        @test PortfolioOptimisers.apply_preprocessing(res, rd) === rd

        # an unfitted split is a pass-through too, so a step that never saw
        # `fit_preprocessing` hands the window on whole rather than cutting it silently
        @test PortfolioOptimisers.apply_preprocessing(tts, rd) === rd
        @test PortfolioOptimisers.apply_preprocessing(tts, pr) === pr

        # it splits whichever data level it is handed
        resp = PortfolioOptimisers.fit_preprocessing(tts, pr)
        @test size(values(resp.train.X), 1) == 80
        @test size(values(resp.test.X), 1) == 20

        # the step writes the sentinel slot and requires no slot up front
        @test PortfolioOptimisers.pipe_writes(tts) == :split
        @test PortfolioOptimisers.pipe_reads(tts) == ()

        # run_step narrows the data slot the input filled, at either level
        ctx = PortfolioOptimisers.PipelineContext(; prices = pr)
        fitted, ctx2 = PortfolioOptimisers.run_step(tts, ctx)
        @test isa(fitted, TrainTestSplitResult)
        @test size(values(ctx2.prices.X), 1) == 80
        @test isnothing(ctx2.returns)

        ctxr = PortfolioOptimisers.PipelineContext(; returns = rd)
        fittedr, ctxr2 = PortfolioOptimisers.run_step(tts, ctxr)
        @test size(fittedr.test.X, 1) == 20
        @test size(ctxr2.returns.X, 1) == 80
    end

    @testset "position rules" begin
        tts = TrainTestSplit(; test_size = 0.2)

        # first step: legal, and auto-named "split"
        pipe = Pipeline(;
                        steps = (tts, PricesToReturns(), EmpiricalPrior(), EqualWeighted()))
        @test pipe.names == ("split", "returns", "prior", "opt")

        # anywhere else: rejected, because an earlier fitted step would have seen the
        # held-out rows
        @test_throws ArgumentError Pipeline(;
                                            steps = (MissingDataFilter(), tts,
                                                     PricesToReturns(), EqualWeighted()))
        @test_throws ArgumentError Pipeline(;
                                            steps = (PricesToReturns(), tts,
                                                     EqualWeighted()))

        # nested inside another pipeline: rejected
        inner = Pipeline(; steps = (tts, PricesToReturns()))
        @test_throws ArgumentError Pipeline(; steps = (inner, EqualWeighted()))

        @test PortfolioOptimisers.has_split(pipe)
        @test !PortfolioOptimisers.has_split(Pipeline(;
                                                      steps = (PricesToReturns(),
                                                               EqualWeighted())))
    end

    @testset "fit, predict and the holdout payoff" begin
        pr = make_prices()

        pipe = Pipeline(;
                        steps = (TrainTestSplit(; test_size = 0.2), MissingDataFilter(),
                                 Imputer(), PricesToReturns(), EmpiricalPrior(),
                                 EqualWeighted()))
        res = fit(pipe, pr)

        # every downstream step saw the training window alone
        @test size(res.ctx.returns.X, 1) == 79   # 80 prices → 79 returns
        @test size(values(res["split"].train.X), 1) == 80
        @test size(values(res["split"].test.X), 1) == 20

        # fit_predict evaluates on the held-out window, not on the training data
        pred = fit_predict(pipe, pr)
        @test length(pred.rd.X) == 19            # 20 held-out prices → 19 returns

        # without a split, fit_predict stays in-sample
        pipe_ns = Pipeline(;
                           steps = (MissingDataFilter(), Imputer(), PricesToReturns(),
                                    EmpiricalPrior(), EqualWeighted()))
        pred_ns = fit_predict(pipe_ns, pr)
        @test length(pred_ns.rd.X) == 99

        # replaying a fitted split on unseen data is a pass-through, so predicting on
        # genuinely new observations still works
        pr_future = make_prices(; T = 130)
        pred_future = predict(res, pr_future, 101:130)
        @test length(pred_future.rd.X) == 29
    end

    @testset "one evaluation protocol per call" begin
        pr = make_prices()
        pipe = Pipeline(;
                        steps = (TrainTestSplit(; test_size = 0.2), PricesToReturns(),
                                 EqualWeighted()))

        # a pipeline carrying a holdout cannot also be cross-validated
        @test_throws ArgumentError PortfolioOptimisers.assert_no_holdout(pipe)

        gscv = GridSearchCrossValidation(Dict("returns" =>
                                                  [PricesToReturns(; ret_method = :simple),
                                                   PricesToReturns(; ret_method = :log)]);
                                         cv = KFold(; n = 2), r = Variance())
        @test_throws ArgumentError search_cross_validation(pipe, gscv, pr)
    end
end
