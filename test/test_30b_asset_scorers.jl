# An Asset Scorer that declares no orientation. The library gives the family no default, so
# `bigger_is_better` on it is a `MethodError`. A struct must be defined at top level.
struct NoOrientationScorer <: PortfolioOptimisers.AbstractAssetScorer end

@testset "Asset scorers" begin
    using Test, PortfolioOptimisers, StableRNGs, Statistics, Clarabel, InteractiveUtils
    PO = PortfolioOptimisers

    rng = StableRNG(19310)
    X = 0.01 .* randn(rng, 120, 6)
    nx = string.('A':'F')
    rd = ReturnsResult(; nx = nx, X = X)
    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false),
                 check_sol = (; allow_local = true, allow_almost = true))

    # The value of `r` on each test window of `cv`, one column for each window.
    function window_values(r, cv, rd)
        wins = PO.distinct_test_windows(split(cv, rd).test_idx)
        return reduce(hcat, [PO.asset_scores(r, rd.X[T, :]) for T in wins])
    end
    # The midrank percentile of each entry of `v`, by hand, with the best entry first.
    function hand_rank(v, bib)
        u = bib ? -v : v
        n = length(u)
        return [(count(<(x), u) + count(<=(x), u)) / (2n) for x in u]
    end

    @testset "the seam: a risk measure is the in-sample score" begin
        @test PO.asset_scores(SCM(), rd) == PO.asset_scores(SCM(), rd.X)
        s = FoldRiskScorer(; r = SCM())
        @test isnothing(PO.assert_scoreable(s))
        @test ScoreSelector(; score = s, rule = RankRule(; best = 2)).score === s
        @test RedundancySelector(; score = s).score === s
        @test_throws MethodError PO.bigger_is_better(NoOrientationScorer())
        for T in subtypes(PO.AbstractAssetScorer)
            T === NoOrientationScorer && continue
            m = which(PO.bigger_is_better, (T,))
            @test m.sig.parameters[2] <: T
        end
    end

    @testset "fold statistics" begin
        v = [0.3, 0.1, 0.1, 0.2]
        @test PO.fold_statistic(Value(), v, false) === v
        @test PO.fold_statistic(Rank(), v, false) == [0.875, 0.25, 0.25, 0.625]
        @test PO.fold_statistic(Rank(), v, false) == hand_rank(v, false)
        @test PO.fold_statistic(Rank(), v, true) == hand_rank(v, true)
        @test PO.fold_statistic(Rank(), v, true) == [0.125, 0.75, 0.75, 0.375]
        @test PO.statistic_bigger_is_better(Value(), true)
        @test !PO.statistic_bigger_is_better(Value(), false)
        @test !PO.statistic_bigger_is_better(Rank(), true)
        @test !PO.statistic_bigger_is_better(Rank(), false)
        @test MeanFoldScore().alg === Value()
        @test MedianFoldScore(; alg = Rank()).alg === Rank()
        @test PO.fold_aggregate(MeanFoldScore(), [1.0, 2.0, 6.0], false) == 3.0
        @test PO.fold_aggregate(MedianFoldScore(), [1.0, 2.0, 6.0], false) == 2.0
        @test PO.fold_aggregate(WorstFoldScore(), [1.0, 2.0, 6.0], false) == 6.0
        @test PO.fold_aggregate(WorstFoldScore(), [1.0, 2.0, 6.0], true) == 1.0
    end

    @testset "FoldRiskScorer equals a hand loop over the test windows" begin
        cv = KFold(; n = 3)
        for r in (SCM(), MeanReturn(), ConditionalValueatRisk())
            bib = PO.bigger_is_better(r)
            V = window_values(r, cv, rd)
            R = reduce(hcat, [hand_rank(c, bib) for c in eachcol(V)])
            worst(M, b) = b ? vec(minimum(M; dims = 2)) : vec(maximum(M; dims = 2))
            cases = ((MeanFoldScore(), vec(mean(V; dims = 2)), bib),
                     (MedianFoldScore(), vec(median(V; dims = 2)), bib),
                     (WorstFoldScore(), worst(V, bib), bib),
                     (MeanFoldScore(; alg = Rank()), vec(mean(R; dims = 2)), false),
                     (MedianFoldScore(; alg = Rank()), vec(median(R; dims = 2)), false),
                     (WorstFoldScore(; alg = Rank()), vec(maximum(R; dims = 2)), false))
            for (agg, expected, ebib) in cases
                s = FoldRiskScorer(; r = r, cv = cv, agg = agg)
                @test PO.asset_scores(s, rd) ≈ expected
                @test PO.bigger_is_better(s) == ebib
            end
        end
        # A risk takes the maximum of its windows, and a reward the minimum.
        V = window_values(MeanReturn(), cv, rd)
        s = FoldRiskScorer(; r = MeanReturn(), cv = cv, agg = WorstFoldScore())
        @test PO.asset_scores(s, rd) == vec(minimum(V; dims = 2))
    end

    @testset "CombinatorialCrossValidation counts each test block once" begin
        ccv = CombinatorialCrossValidation(; n_folds = 4, n_test_folds = 2)
        test_idx = split(ccv, rd).test_idx
        # Six splits of two blocks each hold twelve blocks, and four of them are distinct.
        @test sum(length, test_idx) == 12
        wins = PO.distinct_test_windows(test_idx)
        @test length(wins) == 4
        @test sort(first.(wins)) == [1, 31, 61, 91]
        V = reduce(hcat, [PO.asset_scores(SCM(), rd.X[T, :]) for T in wins])
        s = FoldRiskScorer(; r = SCM(), cv = ccv)
        @test PO.asset_scores(s, rd) ≈ vec(mean(V; dims = 2))
    end

    @testset "FoldRiskScorer scores each window on its Coverage Universe" begin
        X2 = copy(X)
        X2[1:5, 2] .= NaN
        rd2 = ReturnsResult(; nx = nx, X = X2)
        cv = KFold(; n = 3)
        wins = PO.distinct_test_windows(split(cv, rd2).test_idx)
        # Asset B has a gap in the first window alone, so it keeps the other two.
        V = [PO.asset_scores(SCM(), rd2.X[T, setdiff(1:6, k == 1 ? [2] : Int[])])
             for (k, T) in enumerate(wins)]
        s = FoldRiskScorer(; r = SCM(), cv = cv)
        sc = PO.asset_scores(s, rd2)
        @test sc[2] ≈ mean([V[2][2], V[3][2]])
        @test sc[1] ≈ mean([V[1][1], V[2][1], V[3][1]])
        # Under Rank, the first window ranks only the five assets that it covers.
        sr = FoldRiskScorer(; r = SCM(), cv = cv, agg = MeanFoldScore(; alg = Rank()))
        R1 = hand_rank(V[1], false)
        R2 = hand_rank(V[2], false)
        R3 = hand_rank(V[3], false)
        @test PO.asset_scores(sr, rd2)[1] ≈ mean([R1[1], R2[1], R3[1]])
        @test PO.asset_scores(sr, rd2)[2] ≈ mean([R2[2], R3[2]])
        # Two covered windows are fewer than three, and the error names asset B.
        s3 = FoldRiskScorer(; r = SCM(), cv = cv, min_folds = 3)
        err = try
            PO.asset_scores(s3, rd2)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("[\"B\"]", err.msg)
        @test occursin("[2]", err.msg)
        # An asset that no window covers is refused at the default `min_folds = 1`.
        X3 = copy(X)
        X3[1:5, 4] .= NaN
        X3[41:45, 4] .= NaN
        X3[81:85, 4] .= NaN
        @test_throws ArgumentError PO.asset_scores(s, ReturnsResult(; nx = nx, X = X3))
    end

    @testset "FoldRiskScorer validation" begin
        @test_throws ArgumentError FoldRiskScorer(; r = Variance())
        @test_throws DomainError FoldRiskScorer(; r = SCM(), min_folds = 0)
    end

    @testset "WeightMassScorer equals the mean weight over the folds" begin
        iv = InverseVolatility()
        cv = KFold(; n = 3)
        pred = cross_val_predict(iv, rd, cv)
        W = reduce(hcat, [p.res.w for p in pred.pred])
        s = WeightMassScorer(; opt = iv, cv = cv)
        @test PO.asset_scores(s, rd) ≈ vec(mean(W; dims = 2))
        @test PO.bigger_is_better(s)
        # With `tol`, the score is the share of the folds with an absolute weight above it.
        t = median(W)
        st = WeightMassScorer(; opt = iv, cv = cv, tol = t)
        @test PO.asset_scores(st, rd) ≈ vec(mean(abs.(W) .> t; dims = 2))
        @test_throws DomainError WeightMassScorer(; opt = iv, tol = -1.0)
        @test_throws DomainError WeightMassScorer(; opt = iv, min_folds = 0)
    end

    @testset "WeightMassScorer reads each combinatorial split once" begin
        iv = InverseVolatility()
        ccv = CombinatorialCrossValidation(; n_folds = 4, n_test_folds = 2)
        cvr = split(ccv, rd)
        # The weights of each split, fitted once on its training window.
        W = reduce(hcat,
                   [optimise(iv, ReturnsResult(; nx = nx, X = X[tr, :])).w
                    for tr in cvr.train_idx])
        s = WeightMassScorer(; opt = iv, cv = ccv)
        @test PO.asset_scores(s, rd) ≈ vec(mean(W; dims = 2))
    end

    @testset "a grid of two configurations pools both" begin
        sel(k) = ScoreSelector(; score = SCM(), rule = RankRule(; best = k))
        pipe = Pipeline(; steps = ("select" => sel(3), "opt" => InverseVolatility()))
        cv = KFold(; n = 3)
        p = ["select.rule" => [RankRule(; best = 3), RankRule(; best = 5)]]
        s = WeightMassScorer(; opt = pipe, p = p, cv = cv)
        @test length(PO.weight_mass_candidates(s.opt, s.p)) == 2
        # A selector drops assets, and a dropped asset counts with weight zero.
        function full_weights(k)
            pk = Pipeline(; steps = ("select" => sel(k), "opt" => InverseVolatility()))
            pr = cross_val_predict(pk, rd, cv)
            return reduce(hcat,
                          [[get(Dict(zip(q.rd.nx, q.res.w)), n, 0.0) for n in nx]
                           for q in pr.pred])
        end
        W = hcat(full_weights(3), full_weights(5))
        @test PO.asset_scores(s, rd) ≈ vec(mean(W; dims = 2))
        @test any(iszero, full_weights(3))
        @test PO.asset_scores(WeightMassScorer(; opt = pipe, cv = cv), rd) ≈
              vec(mean(full_weights(3); dims = 2))
        # The same grid syntax on an estimator, with its field paths.
        mr(r) = MeanRisk(; r = r, opt = JuMPOptimiser(; slv = slv))
        se = WeightMassScorer(; opt = mr(Variance()), cv = cv,
                              p = ["r" => [Variance(), ConditionalValueatRisk()]])
        We = reduce(hcat,
                    [p.res.w for r in (Variance(), ConditionalValueatRisk())
                     for p in cross_val_predict(mr(r), rd, cv).pred])
        @test PO.asset_scores(se, rd) ≈ vec(mean(We; dims = 2))
    end

    @testset "MultipleRandomised counts only the draws that hold an asset" begin
        iv = InverseVolatility()
        mr = MultipleRandomised(IndexWalkForward(40, 20); subset_size = 3, n_subsets = 3,
                                seed = 7)
        cvr = split(mr, rd)
        held = [count(a -> i in a, cvr.asset_idx) for i in 1:6]
        sums = zeros(6)
        for (tr, a) in zip(cvr.train_idx, cvr.asset_idx)
            w = optimise(iv, ReturnsResult(; nx = nx[a], X = X[tr, a])).w
            sums[a] .+= w
        end
        s = WeightMassScorer(; opt = iv, cv = mr)
        # The seed draws three subsets, and each asset is in at least one of them.
        @test held == [4, 4, 8, 12, 4, 4]
        @test PO.asset_scores(s, rd) ≈ sums ./ held
        k = minimum(held) + 1
        short = nx[held .< k]
        err = try
            PO.asset_scores(WeightMassScorer(; opt = iv, cv = mr, min_folds = k), rd)
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin(string(short), err.msg)
    end

    @testset "WeightMassScorer offers each split its Coverage Universe" begin
        X2 = copy(X)
        X2[1:5, 2] .= NaN
        rd2 = ReturnsResult(; nx = nx, X = X2)
        iv = InverseVolatility()
        cv = KFold(; n = 3)
        cvr = split(cv, rd2)
        # Asset B is in the training window of the first split alone.
        w1 = optimise(iv, ReturnsResult(; nx = nx, X = X2[cvr.train_idx[1], :])).w
        sc = PO.asset_scores(WeightMassScorer(; opt = iv, cv = cv), rd2)
        @test sc[2] ≈ w1[2]
        @test_throws ArgumentError PO.asset_scores(WeightMassScorer(; opt = iv, cv = cv,
                                                                    min_folds = 2), rd2)
    end

    @testset "a failed fold leaves the sum and the count" begin
        N = size(X, 2)
        ok = WeightBounds(; lb = zeros(N), ub = ones(N))
        # `N` assets at a floor of one half cannot sum to one: the fold is infeasible.
        bad = WeightBounds(; lb = fill(0.5, N), ub = ones(N))
        sets = UniverseSets(; dict = Dict("nx" => nx))
        cv = KFold(; n = 3)
        opt = MeanRisk(;
                       opt = JuMPOptimiser(; slv = slv, wb = TimeDependent([ok, bad, ok]),
                                           sets = sets))
        pred = cross_val_predict(opt, rd, cv)
        @test isa(pred.pred[2].res.retcode, OptimisationFailure)
        W = reduce(hcat, [pred.pred[k].res.w for k in (1, 3)])
        s = WeightMassScorer(; opt = opt, cv = cv)
        @test PO.asset_scores(s, rd) ≈ vec(mean(W; dims = 2))
        @test_throws ArgumentError PO.asset_scores(WeightMassScorer(; opt = opt, cv = cv,
                                                                    min_folds = 3), rd)
    end

    @testset "a prediction that the split does not hold is refused" begin
        pred = cross_val_predict(InverseVolatility(), rd, KFold(; n = 3))
        @test_throws ArgumentError PO.prediction_at(pred, 9999)
    end

    @testset "the selectors take a scorer in a Pipeline" begin
        wm = WeightMassScorer(; opt = InverseVolatility(), cv = KFold(; n = 2))
        rule = QuantileRule(; best = 0.5)
        sel = ScoreSelector(; score = wm, rule = rule)
        # The acceptance identity: the public verbs give the score and its orientation.
        @test PO.select_assets(sel, rd) ==
              PO.rule_keep(rule, PO.asset_scores(wm, rd), PO.bigger_is_better(wm))
        fr = FoldRiskScorer(; r = SCM(), cv = KFold(; n = 2))
        red = RedundancySelector(; alg = PairwiseCorrelation(; t = 0.0), score = fr)
        @test count(PO.select_assets(red, rd)) >= 1
        pipe = Pipeline(;
                        steps = ("select" => ScoreSelector(; score = fr,
                                                           rule = RankRule(; best = 3)),
                                 "opt" => InverseVolatility()))
        pr = cross_val_predict(pipe, rd, KFold(; n = 3))
        @test all(p -> length(p.rd.nx) == 3, pr.pred)
        pipe_w = Pipeline(; steps = ("select" => sel, "opt" => EqualWeighted()))
        pw = cross_val_predict(pipe_w, rd, KFold(; n = 3))
        @test all(p -> length(p.rd.nx) == 3, pw.pred)
        pipe_r = Pipeline(; steps = ("select" => red, "opt" => EqualWeighted()))
        @test cross_val_predict(pipe_r, rd, KFold(; n = 3)) isa MultiPeriodPredictionResult
        gscv = GridSearchCrossValidation(["select.score.agg" => [MeanFoldScore(),
                                                                 WorstFoldScore(; alg = Rank())]];
                                         cv = KFold(; n = 3), r = SCM())
        tuned = search_cross_validation(pipe, gscv, rd)
        @test tuned.opt.steps[1].score.agg in
              (MeanFoldScore(), WorstFoldScore(; alg = Rank()))
    end
end
