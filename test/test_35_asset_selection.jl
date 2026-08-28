using Test, PortfolioOptimisers, TimeSeries, Dates, StableRNGs

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

    @testset "tail_mask boundaries" begin
        # the tied block straddles the cut, so ONE asset is admitted, not two and not three
        s = [3.0, 2.0, 2.0, 1.0]
        @test findall(PO.tail_mask(s, 2, true, :best)) == [1]
        @test count(PO.tail_mask(s, 2, true, :best)) == 1

        # the tied block fits inside k, so both members are admitted
        @test findall(PO.tail_mask([3.0, 3.0, 1.0], 2, true, :best)) == [1, 2]

        # the count saturates at both ends
        @test !any(PO.tail_mask(s, 0, true, :best))
        @test !any(PO.tail_mask(s, -1, true, :best))
        @test all(PO.tail_mask(s, 4, true, :best))
        @test all(PO.tail_mask(s, 9, true, :best))

        # bib decides which raw end is `:best`, so the two ends swap under it
        r = [1.0, 2.0, 3.0, 4.0]
        @test PO.tail_mask(r, 2, true, :best) == PO.tail_mask(r, 2, false, :worst)
        @test PO.tail_mask(r, 2, false, :best) == PO.tail_mask(r, 2, true, :worst)
        @test findall(PO.tail_mask(r, 2, true, :best)) == [3, 4]
        @test findall(PO.tail_mask(r, 2, false, :best)) == [1, 2]
        # and the two `:best` masks are complements, because k = 2 halves a 4-asset universe
        @test PO.tail_mask(r, 2, true, :best) == .!PO.tail_mask(r, 2, false, :best)
    end

    @testset "rule boundaries" begin
        s = [1.0, 2.0, 3.0, 4.0, 5.0]

        # ThresholdRule compares strictly, so a score exactly on a bound is DROPPED
        @test findall(PO.rule_keep(ThresholdRule(; lo = 2.0), s, false)) == [3, 4, 5]
        @test findall(PO.rule_keep(ThresholdRule(; hi = 4.0), s, false)) == [1, 2, 3]

        # an omitted bound disables that side alone
        @test findall(PO.rule_keep(ThresholdRule(; hi = 3.0), s, false)) == [1, 2]

        # QuantileRule rounds an exact half UP: 0.05 of ten assets is one, not none
        s10 = collect(1.0:10.0)
        @test count(PO.rule_keep(QuantileRule(; best = 0.05), s10, false)) == 1
        @test count(PO.rule_keep(QuantileRule(; best = 0.04), s10, false)) == 0

        # `:drop` is the exact complement of `:keep` on the same rule and scores
        @test PO.rule_keep(RankRule(; best = 2, action = :drop), s, false) ==
              .!PO.rule_keep(RankRule(; best = 2), s, false)
        @test PO.rule_keep(QuantileRule(; best = 0.4, worst = 0.2, action = :drop), s,
                           false) ==
              .!PO.rule_keep(QuantileRule(; best = 0.4, worst = 0.2), s, false)
    end

    @testset "selection rule validators" begin
        @test_throws PO.IsNothingError PO.assert_tail_counts(nothing, nothing, :RankRule)
        @test_throws DomainError PO.assert_tail_counts(-1, nothing, :RankRule)
        @test_throws DomainError PO.assert_tail_counts(nothing, -1, :RankRule)
        @test_throws DomainError PO.assert_tail_counts(0, 0, :RankRule)
        # a zero on one side is legal: the rule still takes an asset from the other
        @test isnothing(PO.assert_tail_counts(0, 2, :RankRule))
        @test findall(PO.rule_keep(RankRule(; best = 0, worst = 2), [1.0, 2.0, 3.0, 4.0],
                                   false)) == [3, 4]

        @test_throws ArgumentError PO.assert_selection_action(:nonsense)
        @test isnothing(PO.assert_selection_action(:keep))
        @test isnothing(PO.assert_selection_action(:drop))
    end

    @testset "asset_scores and assert_scoreable" begin
        # the score of column i is `score(X[:, i])`, computed one column at a time
        @test PO.asset_scores(SCM(), XZ) == [SCM()(view(XZ, :, i)) for i in axes(XZ, 2)]

        # a NaN score is rejected: Skewness divides by a zero standard deviation
        Xc = [0.1 0.0; -0.1 0.0; 0.2 0.0]
        @test isnan(Skewness()(view(Xc, :, 2)))
        @test_throws DomainError PO.asset_scores(Skewness(), Xc)
        # a drawdown measure on a constant column is finite, so it is not that case
        @test isfinite(MaximumDrawdown()(view(Xc, :, 2)))

        # the message names the wrapper type, and only the two variance measures get the hint
        for m in (Variance(), StandardDeviation())
            e = try
                PO.assert_scoreable(m)
            catch err
                err
            end
            @test occursin(string(Base.typename(typeof(m)).wrapper), e.msg)
            @test occursin("SCM()", e.msg)
        end
        e = try
            PO.assert_scoreable(EqualRisk())
        catch err
            err
        end
        @test occursin("EqualRisk", e.msg)
        @test !occursin("SCM()", e.msg)
    end

    @testset "CompleteAssetSelector drops columns, never observations" begin
        X = [0.1 0.2 NaN 0.4
             0.3 0.2 0.1 0.5
             0.2 0.1 0.2 0.6]
        rd = ReturnsResult(; nx = ["A", "B", "C", "D"], X = X)
        res = fit_preprocessing(CompleteAssetSelector(), rd)
        @test res.nx == ["A", "B", "D"]
        @test size(apply_preprocessing(res, rd).X, 1) == size(X, 1)

        # a `missing` never reaches the selector: a returns carrier binds X to numbers
        @test_throws TypeError ReturnsResult(; nx = ["A", "B"],
                                             X = Union{Missing, Float64}[0.1 0.2
                                                                         0.3 missing])
        # the helper itself does read both sentinels
        @test PO.find_complete_indices(Union{Missing, Float64}[1.0 missing; 2.0 3.0];
                                       dims = 1) == [1]
    end

    @testset "the fitted universe replays" begin
        Xtr = [0.10 0.01 0.001; -0.10 -0.01 -0.001; 0.05 0.02 0.002]
        Xte = [0.001 0.02 0.10; -0.001 -0.02 -0.10; 0.002 0.01 0.05]
        rdtr = ReturnsResult(; nx = ["A", "B", "C"], X = Xtr)
        rdte = ReturnsResult(; nx = ["A", "B", "C"], X = Xte)
        sel = ScoreSelector(; score = SCM(), rule = RankRule(; best = 2))

        res = fit_preprocessing(sel, rdtr)
        @test res.nx == ["B", "C"]
        # the test window's own scores would choose a different universe
        @test fit_preprocessing(sel, rdte).nx == ["A", "B"]
        # the replay keeps the fitted one, which is what makes the selector safe in CV
        @test collect(apply_preprocessing(res, rdte).nx) == ["B", "C"]
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
        res = fit(pipe, pr)
        @test res.ctx.returns.nx == ["A", "B", "D", "E", "F"]
        @test length(res.ctx.opt.w) == 5

        # predict replays the fitted universe on an unseen window
        @test predict(res, pr, 61:120) isa Any

        # a selector after a returns-derived step is rejected at construction
        @test_throws ArgumentError Pipeline(;
                                            steps = (PricesToReturns(), EmpiricalPrior(),
                                                     ZeroVarianceFilter(), EqualWeighted()))
    end
end
