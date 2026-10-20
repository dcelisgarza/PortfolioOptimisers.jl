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
                        steps = ("filter" => MissingDataFilter(),
                                 "gap_fill" => PriceGapFill(), PricesToReturns(),
                                 EmpiricalPrior(), EqualWeighted()))

        # bare step name / symbol / integer address the whole step
        l_name = PortfolioOptimisers.pipeline_lens(pipe, "gap_fill")
        l_sym = PortfolioOptimisers.pipeline_lens(pipe, :gap_fill)
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
        # an indexed string path with no dot is a path too: `Meta.parse` sees the index
        @test PortfolioOptimisers.pipeline_lens(pipe, "steps[1]")(pipe) === pipe.steps[1]
        # a symbol never reaches `Meta.parse`, so an index in one is not structure
        @test_throws ArgumentError PortfolioOptimisers.pipeline_lens(pipe,
                                                                     Symbol("steps[1]"))

        # integer bounds are checked
        @test_throws ArgumentError PortfolioOptimisers.pipeline_lens(pipe, 0)
        @test_throws ArgumentError PortfolioOptimisers.pipeline_lens(pipe, 99)

        # a structureless symbol that is not a step name fails closed rather than
        # silently becoming a property access on the pipeline struct
        @test_throws ArgumentError PortfolioOptimisers.pipeline_lens(pipe, :gap_fill_typo)
        # a dotted symbol is one property name, so `parse_lens` would build a lens on a
        # field of that literal name, which no Pipeline has: it is refused, and the message
        # asks for a String path
        derr = try
            PortfolioOptimisers.pipeline_lens(pipe, Symbol("steps[1].col_thr"))
            nothing
        catch e
            e
        end
        @test derr isa ArgumentError
        @test occursin("write a property path as a `String`", derr.msg)
        # a dotted symbol rooted anywhere else is refused: the root is allowlisted, so the
        # step-name table cannot be addressed
        @test_throws ArgumentError PortfolioOptimisers.pipeline_lens(pipe,
                                                                     Symbol("names.x"))
        @test_throws ArgumentError PortfolioOptimisers.pipeline_lens(pipe,
                                                                     Symbol("names[1].x"))
        # the String arm applies the same rule as its Symbol twin: a structureless key that
        # misses the step-name table is a typo, not a property access on the Pipeline
        @test_throws ArgumentError PortfolioOptimisers.pipeline_lens(pipe, "gap_fill_typo")
        # including one that collides with a real Pipeline field, which used to be written
        # into on every fold
        @test_throws ArgumentError PortfolioOptimisers.pipeline_lens(pipe, "steps")
        @test_throws ArgumentError PortfolioOptimisers.pipeline_lens(pipe, "names")
        # a structured key is not enough: only a path rooted at `steps` is a lens path, so
        # a path into the step-name table is refused rather than written on every fold
        @test_throws ArgumentError PortfolioOptimisers.pipeline_lens(pipe, "names[1]")
        @test_throws ArgumentError PortfolioOptimisers.pipeline_lens(pipe, "names.x")
        @test_throws ArgumentError PortfolioOptimisers.pipeline_lens(pipe, "stepsy[1]")
        # and the refusal names the rule
        rerr = try
            PortfolioOptimisers.pipeline_lens(pipe, "names[1]")
            nothing
        catch e
            e
        end
        @test rerr isa ArgumentError
        @test occursin("nor a property path rooted at `steps`", rerr.msg)
        # and the message names the typo and suggests the step
        serr = try
            PortfolioOptimisers.pipeline_lens(pipe, "gapfill")
            nothing
        catch e
            e
        end
        @test serr isa ArgumentError
        @test occursin("is not a step name", serr.msg)
        @test occursin("did you mean `gap_fill`", serr.msg)
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
                        steps = ("gap_fill" => PriceGapFill(), PricesToReturns(),
                                 EmpiricalPrior(), EqualWeighted()))
        r = ConditionalValueatRisk()
        cv = IndexWalkForward(60, 20)

        # tune the imputation statistic jointly with the workflow
        p = ["gap_fill" =>
                 [PriceGapFill(; fill = MeanValue()), PriceGapFill(; fill = MedianValue())]]
        res = search_cross_validation(pipe, GridSearchCrossValidation(p; cv = cv, r = r),
                                      pr)
        @test res.opt isa Pipeline
        @test size(res.test_scores, 2) == 2
        @test res.opt.steps[1] isa PriceGapFill
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
        # Construct data where the train and test windows disagree about which fill
        # statistic is best. Fitting inside the fold must use train statistics only; the
        # point is that a pipeline never leaks test data.
        rng = StableRNG(2024)
        T, N = 120, 4
        base = 100 .+ cumsum(randn(rng, T, N) / 20; dims = 1)
        # inject a large outlier late in the series (would swing a full-sample mean)
        base[110, 1] = base[110, 1] + 30
        X = TimeArray(make_ts(; T = T), base, string.("A", 1:N))
        vals = copy(values(X))
        vals[5, 1] = NaN
        vals[65, 2] = NaN
        # Both gaps are interior, so the Span Rule bounds every column by the whole clock
        # and the fill reaches them; the fill is bounded by the span, so one is stated.
        pr = PricesResult(; X = TimeArray(timestamp(X), vals, string.("A", 1:N)),
                          span = listing_span(vals))

        pipe = Pipeline(;
                        steps = ("gap_fill" => PriceGapFill(), PricesToReturns(),
                                 EmpiricalPrior(), EqualWeighted()))
        r = ConditionalValueatRisk()
        cv = IndexWalkForward(70, 25)
        p = ["gap_fill" =>
                 [PriceGapFill(; fill = MeanValue()), PriceGapFill(; fill = MedianValue())]]

        res = search_cross_validation(pipe, GridSearchCrossValidation(p; cv = cv, r = r),
                                      pr)
        # the fitted-per-fold scores are finite and the winner is one of the two candidates
        @test all(isfinite, res.test_scores)
        @test res.opt.steps[1] isa PriceGapFill
        @test res.idx in (1, 2)

        # a fold-fitted fill uses train statistics only: refit the winning candidate on
        # the first train window and confirm the fill value came from train
        cvres = split(cv, pr)
        train_idx = cvres.train_idx[1]
        winner = res.opt.steps[1]
        fitted = PortfolioOptimisers.fit_preprocessing(winner,
                                                       PortfolioOptimisers.port_opt_view(pr,
                                                                                         train_idx,
                                                                                         :))
        train_vals = values(PortfolioOptimisers.port_opt_view(pr, train_idx, :).X)[:, 1]
        obs = [x for x in train_vals if !isnan(x)]
        expected = winner.fill isa MeanValue ? mean(obs) : median(obs)
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
                        steps = ("gap_fill" => PriceGapFill(), PricesToReturns(),
                                 EmpiricalPrior(), EqualWeighted()))
        r = ConditionalValueatRisk()
        mr = MultipleRandomised(IndexWalkForward(60, 20); subset_size = 3, n_subsets = 2,
                                seed = 42)
        p = ["gap_fill" =>
                 [PriceGapFill(; fill = MeanValue()), PriceGapFill(; fill = MedianValue())]]

        # the price level must not be rejected (the old rolling-window rule blocked this)
        res = search_cross_validation(pipe, GridSearchCrossValidation(p; cv = mr, r = r),
                                      pr)
        cv = split(mr, pr)
        @test res.opt isa Pipeline
        @test size(res.test_scores, 2) == 2
        @test size(res.test_scores, 1) == length(cv.train_idx)
        @test all(isfinite, res.test_scores)
        @test res.opt.steps[1] isa PriceGapFill
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

    @testset "combinatorial search CV tunes a returns-level pipeline (per-path scoring)" begin
        rd = make_returns()
        pipe = Pipeline(; steps = ("prior" => EmpiricalPrior(), EqualWeighted()))
        r = ConditionalValueatRisk()
        ccv = CombinatorialCrossValidation(; n_folds = 4, n_test_folds = 2)
        p = ["prior" => [EmpiricalPrior(), EmpiricalPrior()]]

        res = search_cross_validation(pipe, GridSearchCrossValidation(p; cv = ccv, r = r),
                                      rd)
        cv = split(ccv, rd)
        n_paths = maximum(cv.path_ids)
        @test res.opt isa Pipeline
        @test size(res.test_scores, 2) == 2                # candidates
        @test size(res.test_scores, 1) == n_paths          # one row per backtest path
        @test all(isfinite, res.test_scores)
        @test res.idx in (1, 2)

        # randomised search delegates to the combinatorial grid
        rs = search_cross_validation(pipe,
                                     RandomisedSearchCrossValidation(p; cv = ccv, r = r,
                                                                     rng = StableRNG(42),
                                                                     n_iter = 2), rd)
        @test size(rs.test_scores) == (n_paths, 2)
        @test all(isfinite, rs.test_scores)

        # combinatorial now runs at the price level too (boundary-return approximation over
        # the non-contiguous training rows); test groups are contiguous so scoring holds
        X = make_prices()
        pr = PricesResult(; X = X)
        pipe_pr = Pipeline(;
                           steps = ("gap_fill" => PriceGapFill(), PricesToReturns(),
                                    EmpiricalPrior(), EqualWeighted()))
        p_pr = ["gap_fill" => [PriceGapFill(; fill = MeanValue()),
                               PriceGapFill(; fill = MedianValue())]]
        res_pr = search_cross_validation(pipe_pr,
                                         GridSearchCrossValidation(p_pr; cv = ccv, r = r),
                                         pr)
        @test res_pr.opt isa Pipeline
        @test size(res_pr.test_scores) == (maximum(split(ccv, pr).path_ids), 2)
        @test all(isfinite, res_pr.test_scores)
    end

    @testset "an Expr key is rooted at `steps`, and a prebuilt lens passes through" begin
        pipe = Pipeline(;
                        steps = ("filter" => MissingDataFilter(),
                                 "gap_fill" => PriceGapFill(), PricesToReturns(),
                                 EmpiricalPrior(), EqualWeighted()))
        l = PortfolioOptimisers.pipeline_lens(pipe, :(steps[1].col_thr))
        @test l(pipe) == pipe.steps[1].col_thr
        @test Accessors.set(pipe, l, 0.5).steps[1].col_thr == 0.5
        # the String arm refuses a path into the step-name table, and so does the Expr arm:
        # a search wrote the grid value into `names` on every fold
        @test_throws ArgumentError PortfolioOptimisers.pipeline_lens(pipe, :(names[1]))
        @test_throws ArgumentError PortfolioOptimisers.pipeline_lens(pipe,
                                                                     :(filter.col_thr))
        pre = PortfolioOptimisers.parse_lens("steps[2].fill")
        @test PortfolioOptimisers.pipeline_lens(pipe, pre) === pre
        # a dotted Symbol key reaches the grid as a refusal, not as an error from the
        # lens setter inside the candidate loop
        rd = make_returns()
        pipe2 = Pipeline(; steps = ("prior" => EmpiricalPrior(), EqualWeighted()))
        @test_throws ArgumentError search_cross_validation(pipe2,
                                                           GridSearchCrossValidation([Symbol("steps[1].ce") =>
                                                                                          [PortfolioOptimisersCovariance()]];
                                                                                     cv = IndexWalkForward(60,
                                                                                                           20),
                                                                                     r = ConditionalValueatRisk()),
                                                           rd)
    end

    @testset "pipeline_lens_val_grid builds the grid of lens_val_grid" begin
        pipe = Pipeline(;
                        steps = ("filter" => MissingDataFilter(),
                                 "gap_fill" => PriceGapFill(), PricesToReturns(),
                                 EmpiricalPrior(), EqualWeighted()))
        fills = [PriceGapFill(; fill = MeanValue()), PriceGapFill(; fill = MedianValue())]
        # one set of pairs: the product, first key fastest, and the same grid as the raw
        # paths give to `lens_val_grid`
        lenses, vals = PortfolioOptimisers.pipeline_lens_val_grid(pipe,
                                                                  ["filter.col_thr" =>
                                                                       [0.5, 0.6, 0.7],
                                                                   "gap_fill" => fills])
        rl, rv = PortfolioOptimisers.lens_val_grid(["steps[1].col_thr" => [0.5, 0.6, 0.7],
                                                    "steps[2]" => fills])
        @test length(vals) == 6
        @test vals == rv
        @test vals[1] == (0.5, fills[1]) &&
              vals[2] == (0.6, fills[1]) &&
              vals[4] == (0.5, fills[2])
        for k in eachindex(vals)
            @test [l(pipe) for l in lenses[k]] == [l(pipe) for l in rl[k]]
        end
        # a dictionary
        dl, dv = PortfolioOptimisers.pipeline_lens_val_grid(pipe,
                                                            Dict("filter.col_thr" =>
                                                                     [0.5, 0.6]))
        @test dv == [(0.5,), (0.6,)]
        @test dl[1][1](pipe) == pipe.steps[1].col_thr
        # a vector of sets concatenates, so the sizes add
        sl, sv = PortfolioOptimisers.pipeline_lens_val_grid(pipe,
                                                            Union{Vector{Pair{String,
                                                                              Vector{Float64}}},
                                                                  Dict{String,
                                                                       typeof(fills)}}[["filter.col_thr" =>
                                                                                            [0.5,
                                                                                             0.6]],
                                                                                       Dict("gap_fill" =>
                                                                                                fills)])
        @test length(sv) == length(sl) == 2 + 2
        @test sv[3] == (fills[1],)
        @test sl[3][1](pipe) === pipe.steps[2]
        # an empty value vector and a grid over the cap are refused
        @test_throws PortfolioOptimisers.IsEmptyError PortfolioOptimisers.pipeline_lens_val_grid(pipe,
                                                                                                 ["filter.col_thr" =>
                                                                                                      Float64[]])
        PortfolioOptimisers.with_resource_limits(; max_search_grid = 5) do
            @test_throws DomainError PortfolioOptimisers.pipeline_lens_val_grid(pipe,
                                                                                ["filter.col_thr" =>
                                                                                     [0.5,
                                                                                      0.6,
                                                                                      0.7],
                                                                                 "gap_fill" =>
                                                                                     fills])
            @test_throws DomainError PortfolioOptimisers.pipeline_lens_val_grid(pipe,
                                                                                [["filter.col_thr" =>
                                                                                      [0.5,
                                                                                       0.6,
                                                                                       0.7]],
                                                                                 ["filter.col_thr" =>
                                                                                      [0.5,
                                                                                       0.6,
                                                                                       0.7]]])
        end
    end

    @testset "the views and the score type of a Pipeline search" begin
        pr = PricesResult(; X = make_prices())
        rd = make_returns()
        @test values(PortfolioOptimisers.pipeline_asset_view(pr, [2, 4]).X) ==
              values(pr.X)[:, [2, 4]]
        @test PortfolioOptimisers.pipeline_asset_view(rd, [2, 4]).X == rd.X[:, [2, 4]]
        @test values(PortfolioOptimisers.pipeline_data_view(pr, 11:20).X) ==
              values(pr.X)[11:20, :]
        @test PortfolioOptimisers.pipeline_data_view(rd, 11:20, [1, 3]).X ==
              rd.X[11:20, [1, 3]]
        @test PortfolioOptimisers.cv_data_eltype(rd) === Float64
        @test PortfolioOptimisers.cv_data_eltype(ReturnsResult(; nx = rd.nx,
                                                               X = Float32.(rd.X))) ===
              Float32
        # integer prices: a score is a fraction and a failed fold scores NaN, so the score
        # matrix is floating point, and the search agrees with the same prices as Float64
        Xi = round.(Int, 100 .+ cumsum(randn(StableRNG(1), 120, 5); dims = 1))
        ts = make_ts()
        pri = PricesResult(; X = TimeArray(ts, Xi, string.("A", 1:5)))
        prf = PricesResult(; X = TimeArray(ts, float.(Xi), string.("A", 1:5)))
        @test PortfolioOptimisers.cv_data_eltype(pri) === Float64
        pipe = Pipeline(;
                        steps = (PricesToReturns(), "prior" => EmpiricalPrior(),
                                 EqualWeighted()))
        gscv = GridSearchCrossValidation(["opt" => [EqualWeighted(), InverseVolatility()]];
                                         cv = IndexWalkForward(60, 20),
                                         r = ConditionalValueatRisk(), train_score = true)
        resi = search_cross_validation(pipe, gscv, pri)
        resf = search_cross_validation(pipe, gscv, prf)
        @test eltype(resi.test_scores) === Float64
        @test resi.test_scores ≈ resf.test_scores
        @test resi.idx == resf.idx
        cgscv = GridSearchCrossValidation(["opt" => [EqualWeighted(), InverseVolatility()]];
                                          cv = CombinatorialCrossValidation(; n_folds = 4,
                                                                            n_test_folds = 2),
                                          r = ConditionalValueatRisk(), train_score = true)
        cresi = search_cross_validation(pipe, cgscv, pri)
        cresf = search_cross_validation(pipe, cgscv, prf)
        @test eltype(cresi.test_scores) === Float64
        @test cresi.test_scores ≈ cresf.test_scores
    end

    @testset "the scores, the train scores and the winner, checked against the fold loop" begin
        rd = make_returns()
        pr = PricesResult(; X = make_prices())
        r = ConditionalValueatRisk()
        l = PortfolioOptimisers.parse_lens("steps[2]")
        cands = [EqualWeighted(), InverseVolatility()]
        # a grid search at the price level: S_fi = -CVaR of fold f, since CVaR is not
        # bigger-is-better, and the winner is the column of greatest mean
        ppipe = Pipeline(;
                         steps = (PricesToReturns(), EmpiricalPrior(),
                                  "opt" => EqualWeighted()))
        pl = PortfolioOptimisers.parse_lens("steps[3]")
        cv = IndexWalkForward(60, 20)
        res = search_cross_validation(ppipe,
                                      GridSearchCrossValidation(["opt" => cands]; cv = cv,
                                                                r = r, train_score = true),
                                      pr)
        for (i, c) in enumerate(cands)
            loop = cross_val_predict(Accessors.set(ppipe, pl, c), pr, cv)
            @test res.test_scores[:, i] == [-expected_risk(r, p) for p in loop.pred]
            @test res.train_scores[:, i] == [-expected_risk(r, p.res) for p in loop.pred]
        end
        @test res.idx == argmax(vec(mean(res.test_scores; dims = 1)))
        @test res.opt.steps[3] === cands[res.idx]
        # a combinatorial search: one score per path, and one train score per fold of a path
        pipe = Pipeline(; steps = ("prior" => EmpiricalPrior(), "opt" => EqualWeighted()))
        ccv = CombinatorialCrossValidation(; n_folds = 4, n_test_folds = 2)
        cres = search_cross_validation(pipe,
                                       GridSearchCrossValidation(["opt" => cands]; cv = ccv,
                                                                 r = r, train_score = true),
                                       rd)
        n_paths = maximum(split(ccv, rd).path_ids)
        @test length(cres.train_scores) == n_paths
        for (i, c) in enumerate(cands)
            preds = cross_val_predict(Accessors.set(pipe, l, c), rd, ccv)
            @test cres.test_scores[:, i] == -expected_risk(r, preds)
            for (p, path) in enumerate(preds.pred)
                @test cres.train_scores[p][:, i] ==
                      [-expected_risk(r, fp.res) for fp in path.pred]
            end
        end
        @test cres.idx == argmax(vec(mean(cres.test_scores; dims = 1)))
        @test cres.opt.steps[2] === cands[cres.idx]
    end

    @testset "the combinatorial search checks its entry like the grid search" begin
        rd = make_returns()
        r = ConditionalValueatRisk()
        ccv = CombinatorialCrossValidation(; n_folds = 4, n_test_folds = 2)
        cands = [EqualWeighted(), InverseVolatility()]
        # a warm pipeline carries a partial-fit state that no fold reads
        warm = PortfolioOptimisers.partial_fit!(Pipeline(;
                                                         steps = (EmpiricalPrior(),
                                                                  EqualWeighted())),
                                                PortfolioOptimisers.port_opt_view(rd, 1:60,
                                                                                  :))
        for scheme in (IndexWalkForward(60, 20), ccv)
            @test_throws ArgumentError search_cross_validation(warm,
                                                               GridSearchCrossValidation(["steps[2]" =>
                                                                                              cands];
                                                                                         cv = scheme,
                                                                                         r = r),
                                                               rd)
            # a lens that writes a holdout into a candidate is refused before the parallel
            # loop, so the error is the refusal itself and not a TaskFailedException
            pipe = Pipeline(; steps = ("prior" => EmpiricalPrior(), EqualWeighted()))
            @test_throws ArgumentError search_cross_validation(pipe,
                                                               GridSearchCrossValidation(["prior" =>
                                                                                              [EmpiricalPrior(),
                                                                                               TrainTestSplit(;
                                                                                                              test_size = 0.2)]];
                                                                                         cv = scheme,
                                                                                         r = r),
                                                               rd)
        end
    end
end
