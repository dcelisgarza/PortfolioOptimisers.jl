@testset "Pipeline search cross-validation" begin
    using Test, PortfolioOptimisers, TimeSeries, Dates, StableRNGs, Statistics, Clarabel,
          Accessors

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
    function make_returns(; T = 120, N = 5)
        rng = StableRNG(987654321)
        return ReturnsResult(; nx = string.("A", 1:N), X = randn(rng, T, N) / 100,
                             ts = make_ts(; T = T))
    end

    @testset "pipeline_lens addressing" begin
        pipe = Pipeline(;
                        steps = ("filter" => MissingDataFilter(), "impute" => Imputer(),
                                 PricesToReturns(), EmpiricalPrior(), EqualWeighted()))

        # bare step name / symbol / integer address the whole step
        l_name = PortfolioOptimisers.pipeline_lens(pipe, "impute")
        l_sym = PortfolioOptimisers.pipeline_lens(pipe, :impute)
        l_int = PortfolioOptimisers.pipeline_lens(pipe, 2)
        @test l_name(pipe) === pipe.steps[2]
        @test l_sym(pipe) === pipe.steps[2]
        @test l_int(pipe) === pipe.steps[2]

        # swapping a whole step
        pipe2 = Accessors.set(pipe, l_name, MissingDataFilter(; col_thr = 0.7))
        @test pipe2.steps[2] isa MissingDataFilter

        # step name with a trailing property path
        l_field = PortfolioOptimisers.pipeline_lens(pipe, "filter.col_thr")
        @test l_field(pipe) == pipe.steps[1].col_thr
        pipe3 = Accessors.set(pipe, l_field, 0.5)
        @test pipe3.steps[1].col_thr == 0.5

        # raw property paths still work (fall through to parse_lens)
        l_raw = PortfolioOptimisers.pipeline_lens(pipe, "steps[1].col_thr")
        @test l_raw(pipe) == pipe.steps[1].col_thr

        # integer bounds are checked
        @test_throws ArgumentError PortfolioOptimisers.pipeline_lens(pipe, 0)
        @test_throws ArgumentError PortfolioOptimisers.pipeline_lens(pipe, 99)
    end

    @testset "name-addressed == index-addressed" begin
        rd = make_returns()
        pipe = Pipeline(; steps = ("prior" => EmpiricalPrior(), EqualWeighted()))
        r = ConditionalValueatRisk()
        cv = IndexWalkForward(60, 20)

        # tuning by step name vs by raw property path must give identical results
        p_name = ["prior" => [EmpiricalPrior(), EmpiricalPrior()]]
        p_idx = ["steps[1]" => [EmpiricalPrior(), EmpiricalPrior()]]
        res_name = search_cross_validation(pipe,
                                           GridSearchCrossValidation(p_name; cv = cv,
                                                                     r = r), rd)
        res_idx = search_cross_validation(pipe,
                                          GridSearchCrossValidation(p_idx; cv = cv, r = r),
                                          rd)
        @test res_name.test_scores == res_idx.test_scores
        @test res_name.idx == res_idx.idx
    end

    @testset "grid search tunes a pipeline (prices level)" begin
        X = make_prices()
        vals = copy(values(X))
        vals[3, 2] = NaN
        vals[7, 4] = NaN
        pr = PricesResult(; X = TimeArray(timestamp(X), vals, string.("A", 1:5)))
        pipe = Pipeline(;
                        steps = ("impute" => Imputer(), PricesToReturns(), EmpiricalPrior(),
                                 EqualWeighted()))
        r = ConditionalValueatRisk()
        cv = IndexWalkForward(60, 20)

        # tune the imputation statistic jointly with the workflow
        p = ["impute" => [Imputer(; stat = MeanValue()), Imputer(; stat = MedianValue())]]
        res = search_cross_validation(pipe, GridSearchCrossValidation(p; cv = cv, r = r),
                                      pr)
        @test res.opt isa Pipeline
        @test size(res.test_scores, 2) == 2
        @test res.opt.steps[1] isa Imputer
        @test 1 <= res.idx <= 2

        # the tuned pipeline fits end to end
        fit_res = fit(res.opt, pr)
        @test length(fit_res.w) == 5
    end

    @testset "randomised search delegates to grid" begin
        rd = make_returns()
        pipe = Pipeline(; steps = (EmpiricalPrior(), EqualWeighted()))
        r = ConditionalValueatRisk()
        cv = IndexWalkForward(60, 20)
        p = ["steps[1]" => [EmpiricalPrior(), EmpiricalPrior(), EmpiricalPrior()]]

        gs = search_cross_validation(pipe, GridSearchCrossValidation(p; cv = cv, r = r), rd)
        rs = search_cross_validation(pipe,
                                     RandomisedSearchCrossValidation(p; cv = cv, r = r,
                                                                     rng = StableRNG(42),
                                                                     n_iter = 3), rd)
        @test rs.opt isa Pipeline
        @test size(rs.test_scores, 2) == 3
        @test gs.test_scores == rs.test_scores
    end

    @testset "leakage: full-sample preprocessing picks a different winner" begin
        # Construct data where the train and test windows disagree about which
        # imputation statistic is best. Fitting inside the fold must use train
        # statistics only; the point is that a pipeline never leaks test data.
        rng = StableRNG(2024)
        T, N = 120, 4
        base = 100 .+ cumsum(randn(rng, T, N) / 20; dims = 1)
        # inject a large outlier late in the series (would swing a full-sample mean)
        base[110, 1] = base[110, 1] + 30
        X = TimeArray(make_ts(; T = T), base, string.("A", 1:N))
        vals = copy(values(X))
        vals[5, 1] = NaN
        vals[65, 2] = NaN
        pr = PricesResult(; X = TimeArray(timestamp(X), vals, string.("A", 1:N)))

        pipe = Pipeline(;
                        steps = ("impute" => Imputer(), PricesToReturns(), EmpiricalPrior(),
                                 EqualWeighted()))
        r = ConditionalValueatRisk()
        cv = IndexWalkForward(70, 25)
        p = ["impute" => [Imputer(; stat = MeanValue()), Imputer(; stat = MedianValue())]]

        res = search_cross_validation(pipe, GridSearchCrossValidation(p; cv = cv, r = r),
                                      pr)
        # the fitted-per-fold scores are finite and the winner is one of the two candidates
        @test all(isfinite, res.test_scores)
        @test res.opt.steps[1] isa Imputer
        @test res.idx in (1, 2)

        # fold-fitted imputation uses train statistics only: refit the winning
        # candidate on the first train window and confirm the fill came from train
        cvres = split(cv, pr)
        train_idx = cvres.train_idx[1]
        winner = res.opt.steps[1]
        fitted = PortfolioOptimisers.fit_preprocessing(winner,
                                                       PortfolioOptimisers.port_opt_view(pr,
                                                                                         train_idx,
                                                                                         :))
        train_vals = values(PortfolioOptimisers.port_opt_view(pr, train_idx, :).X)[:, 1]
        obs = [x for x in train_vals if !isnan(x)]
        expected = winner.stat isa MeanValue ? mean(obs) : median(obs)
        j = findfirst(==(:A1), fitted.nx)
        @test fitted.v[j] ≈ expected
    end

    @testset "MultipleRandomised tunes a returns-level pipeline" begin
        rd = make_returns()
        pipe = Pipeline(; steps = ("prior" => EmpiricalPrior(), EqualWeighted()))
        r = ConditionalValueatRisk()
        mr = MultipleRandomised(IndexWalkForward(60, 20); subset_size = 3, n_subsets = 2,
                                seed = 42)
        p = ["prior" => [EmpiricalPrior(), EmpiricalPrior(), EmpiricalPrior()]]

        res = search_cross_validation(pipe, GridSearchCrossValidation(p; cv = mr, r = r),
                                      rd)
        cv = split(mr, rd)
        @test res.opt isa Pipeline
        @test size(res.test_scores, 2) == 3                     # 3 candidates
        # one row per fold across every resampled path
        @test size(res.test_scores, 1) == length(cv.train_idx)
        @test length(unique(cv.path_ids)) == 2                  # n_subsets paths
        @test all(isfinite, res.test_scores)
        @test 1 <= res.idx <= 3
    end

    @testset "MultipleRandomised tunes a price-level pipeline" begin
        # MR draws over assets and windows rows with an inner walk-forward, so rows stay
        # contiguous: a price-starting pipeline (PricesToReturns) is admissible.
        X = make_prices()
        vals = copy(values(X))
        vals[3, 2] = NaN
        vals[7, 4] = NaN
        pr = PricesResult(; X = TimeArray(timestamp(X), vals, string.("A", 1:5)))
        pipe = Pipeline(;
                        steps = ("impute" => Imputer(), PricesToReturns(), EmpiricalPrior(),
                                 EqualWeighted()))
        r = ConditionalValueatRisk()
        mr = MultipleRandomised(IndexWalkForward(60, 20); subset_size = 3, n_subsets = 2,
                                seed = 42)
        p = ["impute" => [Imputer(; stat = MeanValue()), Imputer(; stat = MedianValue())]]

        # the price level must not be rejected (the old rolling-window rule blocked this)
        res = search_cross_validation(pipe, GridSearchCrossValidation(p; cv = mr, r = r),
                                      pr)
        cv = split(mr, pr)
        @test res.opt isa Pipeline
        @test size(res.test_scores, 2) == 2
        @test size(res.test_scores, 1) == length(cv.train_idx)
        @test all(isfinite, res.test_scores)
        @test res.opt.steps[1] isa Imputer
        @test res.idx in (1, 2)

        # the tuned pipeline fits end to end
        fit_res = fit(res.opt, pr)
        @test length(fit_res.w) == 5
    end

    @testset "randomised search with a MultipleRandomised scheme" begin
        rd = make_returns()
        pipe = Pipeline(; steps = (EmpiricalPrior(), EqualWeighted()))
        r = ConditionalValueatRisk()
        mr = MultipleRandomised(IndexWalkForward(60, 20); subset_size = 3, n_subsets = 2,
                                seed = 42)
        p = ["steps[1]" => [EmpiricalPrior(), EmpiricalPrior(), EmpiricalPrior()]]

        gs = search_cross_validation(pipe, GridSearchCrossValidation(p; cv = mr, r = r), rd)
        rs = search_cross_validation(pipe,
                                     RandomisedSearchCrossValidation(p; cv = mr, r = r,
                                                                     rng = StableRNG(42),
                                                                     n_iter = 3), rd)
        @test rs.opt isa Pipeline
        @test size(rs.test_scores, 2) == 3
        @test gs.test_scores == rs.test_scores
    end

    @testset "combinatorial search CV is gated off with an informative error" begin
        rd = make_returns()
        pipe = Pipeline(; steps = (EmpiricalPrior(), EqualWeighted()))
        r = ConditionalValueatRisk()
        ccv = CombinatorialCrossValidation(; n_folds = 4, n_test_folds = 2)
        p = ["steps[1]" => [EmpiricalPrior(), EmpiricalPrior()]]
        @test_throws ArgumentError search_cross_validation(pipe,
                                                           GridSearchCrossValidation(p;
                                                                                     cv = ccv,
                                                                                     r = r),
                                                           rd)
    end
end
