using Test, PortfolioOptimisers, TimeSeries, Dates, StableRNGs, StatsAPI

# a selector that never implements select_assets, for the erroring-fallback test
struct UnimplementedSelector <: PortfolioOptimisers.AbstractAssetSelector end

@testset "Asset selection" begin
    PO = PortfolioOptimisers

    # A: var 0.0233, B: constant, C: var 0.07, D: 3.3e-19
    XZ = [0.1 0.0 -0.2 0.0
          -0.1 0.0 0.3 1e-9
          0.2 0.0 -0.1 0.0]
    RDZ = ReturnsResult(; nx = ["A", "B", "C", "D"], X = XZ)

    @testset "rule construction validation" begin
        @test_throws PO.IsNothingError ThresholdRule()
        @test_throws DomainError ThresholdRule(; lo = 1.0, hi = 0.0)

        @test_throws PO.IsNothingError RankRule()
        @test_throws DomainError RankRule(; best = -1)
        @test_throws DomainError RankRule(; best = 0, worst = 0)
        @test_throws ArgumentError RankRule(; best = 1, action = :nonsense)

        @test_throws PO.IsNothingError QuantileRule()
        @test_throws DomainError QuantileRule(; best = 0.0)
        @test_throws DomainError QuantileRule(; worst = 1.0)
        @test_throws ArgumentError QuantileRule(; best = 0.5, action = :nonsense)
    end

    @testset "rule semantics" begin
        s = [1.0, 2.0, 3.0, 4.0, 5.0]

        # ordinal rules read orientation from bigger_is_better
        @test findall(PO.rule_keep(RankRule(; best = 2), s, false)) == [1, 2]
        @test findall(PO.rule_keep(RankRule(; best = 2), s, true)) == [4, 5]
        @test findall(PO.rule_keep(RankRule(; worst = 2), s, false)) == [4, 5]

        # action complements the whole selection
        @test findall(PO.rule_keep(RankRule(; worst = 1, action = :drop), s, false)) ==
              [1, 2, 3, 4]

        # both tails at once; :drop keeps the middle
        @test findall(PO.rule_keep(RankRule(; best = 1, worst = 1), s, false)) == [1, 5]
        @test findall(PO.rule_keep(RankRule(; best = 1, worst = 1, action = :drop), s,
                                   false)) == [2, 3, 4]

        # counts saturate rather than throwing, so a grid search over k survives
        @test all(PO.rule_keep(RankRule(; best = 99), s, false))

        # quantile bounds are fractions of the universe
        @test findall(PO.rule_keep(QuantileRule(; best = 0.4), s, false)) == [1, 2]
        @test findall(PO.rule_keep(QuantileRule(; worst = 0.2, action = :drop), s, false)) ==
              [1, 2, 3, 4]

        # thresholds are literal: they ignore bigger_is_better
        @test findall(PO.rule_keep(ThresholdRule(; lo = 1.5, hi = 4.5), s, false)) ==
              [2, 3, 4]
        @test PO.rule_keep(ThresholdRule(; lo = 1.5), s, false) ==
              PO.rule_keep(ThresholdRule(; lo = 1.5), s, true)

        # bounds are exclusive
        @test findall(PO.rule_keep(ThresholdRule(; lo = 1.0), s, false)) == [2, 3, 4, 5]
    end

    @testset "tie policy: excluded, never split" begin
        # a tied block straddling the cut is dropped entirely, so fewer than k survive
        @test findall(PO.rule_keep(RankRule(; best = 2), [1.0, 2.0, 2.0, 3.0], false)) ==
              [1]

        # a tied block that fits within k is kept whole
        @test findall(PO.rule_keep(RankRule(; best = 2), [1.0, 1.0, 3.0, 4.0], false)) ==
              [1, 2]

        # nothing is distinguishable, so nothing is selected
        @test !any(PO.rule_keep(RankRule(; best = 2), [7.0, 7.0, 7.0], false))

        # and an empty selection fails closed at fit time
        rd = ReturnsResult(; nx = ["A", "B", "C"], X = [1.0 1.0 1.0; -1.0 -1.0 -1.0])
        @test_throws PO.IsEmptyError fit_preprocessing(ScoreSelector(; score = SCM(),
                                                                     rule = RankRule(;
                                                                                     best = 2)),
                                                       rd)
    end

    @testset "ScoreSelector" begin
        # Variance is a WeightsInput measure: it cannot score one asset's return series
        @test_throws ArgumentError ScoreSelector(; score = Variance(),
                                                 rule = ThresholdRule(; lo = 0.0))
        @test_throws ArgumentError ScoreSelector(; score = StandardDeviation(),
                                                 rule = ThresholdRule(; lo = 0.0))
        err = try
            ScoreSelector(; score = Variance(), rule = ThresholdRule(; lo = 0.0))
        catch e
            e
        end
        @test occursin("SCM()", err.msg)

        # SCM scores each column's variance
        @test PO.asset_scores(SCM(), XZ) ≈
              [0.02333333333333333, 0.0, 0.07, 3.3333333333333335e-19]

        # keep the two highest-variance assets
        sel = ScoreSelector(; score = SCM(), rule = RankRule(; best = 2, action = :drop))
        @test fit_preprocessing(sel, RDZ).nx == ["A", "C"]

        # MeanReturn is maximised, so :best is the highest mean
        rd = ReturnsResult(; nx = ["A", "B", "C"], X = [0.1 0.2 0.3; 0.1 0.2 0.3])
        @test fit_preprocessing(ScoreSelector(; score = MeanReturn(),
                                              rule = RankRule(; best = 1)), rd).nx == ["C"]
        @test fit_preprocessing(ScoreSelector(; score = MeanReturn(),
                                              rule = RankRule(; worst = 1)), rd).nx == ["A"]

        # non-finite scores are rejected rather than sorted arbitrarily
        @test_throws DomainError PO.asset_scores(MaximumDrawdown(),
                                                 [NaN 0.1; NaN 0.2; NaN -0.1])
    end

    @testset "ZeroVarianceFilter alias" begin
        @test ZeroVarianceFilter() isa ScoreSelector
        @test fit_preprocessing(ZeroVarianceFilter(), RDZ).nx == ["A", "C"]

        # the bound is exclusive, so tol = 0 still drops an exactly-constant asset
        @test fit_preprocessing(ZeroVarianceFilter(; tol = 0.0), RDZ).nx == ["A", "C", "D"]
        @test_throws DomainError ZeroVarianceFilter(; tol = -1.0)
    end

    @testset "CompleteAssetSelector" begin
        rd = ReturnsResult(; nx = ["A", "B"], X = [0.1 0.2; 0.3 NaN])
        @test fit_preprocessing(CompleteAssetSelector(), rd).nx == ["A"]
        @test_throws PO.IsEmptyError fit_preprocessing(CompleteAssetSelector(),
                                                       ReturnsResult(; nx = ["A"],
                                                                     X = [0.1; NaN;;]))
    end

    @testset "fit/apply contract" begin
        res = fit_preprocessing(ZeroVarianceFilter(), RDZ)
        @test res isa AssetSelectorResult
        @test res isa PortfolioOptimisers.AbstractReturnsPreprocessingResult

        # apply sub-selects by name, in fitted order, even when the window reorders columns
        shuffled = ReturnsResult(; nx = ["C", "D", "B", "A"], X = XZ[:, [3, 4, 2, 1]])
        @test collect(apply_preprocessing(res, shuffled).nx) == ["A", "C"]

        # a fitted asset absent from the window throws rather than shrinking the universe
        partial = ReturnsResult(; nx = ["A", "B"], X = XZ[:, 1:2])
        @test_throws ArgumentError apply_preprocessing(res, partial)

        # extra assets in the window are simply not selected
        extra = ReturnsResult(; nx = ["A", "B", "C", "D", "E"],
                              X = hcat(XZ, [1.0; 2.0; 3.0]))
        @test collect(apply_preprocessing(res, extra).nx) == ["A", "C"]

        # the erroring fallback names the missing method
        @test_throws ArgumentError fit_preprocessing(UnimplementedSelector(), RDZ)
    end

    @testset "pipeline integration" begin
        rng = StableRNG(1234)
        T, N = 120, 6
        ts = collect(Date(2020, 1, 1):Day(1):(Date(2020, 1, 1) + Day(T - 1)))
        Xp = 100 .+ cumsum(0.01 .* randn(rng, T, N); dims = 1)
        Xp[:, 3] .= 100.0                      # C is constant
        pr = PricesResult(; X = TimeArray(ts, Xp, ["A", "B", "C", "D", "E", "F"]))

        # selectors are steppable with no pipeline-side code
        @test PortfolioOptimisers.pipe_writes(ZeroVarianceFilter()) == :returns
        @test PortfolioOptimisers.pipe_reads(ZeroVarianceFilter()) == (:returns,)

        pipe = Pipeline(;
                        steps = (MissingDataFilter(), Imputer(), PricesToReturns(),
                                 ZeroVarianceFilter(), EmpiricalPrior(), EqualWeighted()))
        res = StatsAPI.fit(pipe, pr)
        @test res.ctx.returns.nx == ["A", "B", "D", "E", "F"]
        @test length(res.ctx.opt.w) == 5

        # predict replays the fitted universe on an unseen window
        @test StatsAPI.predict(res, pr, 61:120) isa Any

        # a selector after a returns-derived step is rejected at construction
        @test_throws ArgumentError Pipeline(;
                                            steps = (PricesToReturns(), EmpiricalPrior(),
                                                     ZeroVarianceFilter(), EqualWeighted()))
    end
end
