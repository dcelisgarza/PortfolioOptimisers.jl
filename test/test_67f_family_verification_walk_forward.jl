#=
The checks of this subject are split over the files `test_67f_family_verification*.jl`, so that each file of the
suite runs in about the same time. This file holds one part of them.

The online portfolio selection family, verified end to end over its whole roster (issue
#1179 on map #1148). The per-set files prove each rule's own contracts; this file runs every
rule of the family through the same six checks, so a rule added later joins the roster here
and is held to the family's promises:

 1. the batch-online identity is exact at every block size, on a static panel and on a
    time-varying one, and `Resume` re-enters every rule;
 2. every rule steps on a `ProgrammeAllocationSet` with a turnover ceiling and a cardinality
    bound, and honours both;
 3. every rule runs a fee-charged, drifted walk-forward, and the hindsight comparators and
    `log_wealth_regret` run against each;
 4. the two-regime diagnostic of the user guide reproduces on the library's own fixture;
 5. the prototype's corrections hold on the shipped code;
 6. a weight path renders through the walk-forward's plots and a fold's read-out reaches the
    finite allocation unchanged.
=#

@testset "Online portfolio selection: the family end to end" begin
    using Test, PortfolioOptimisers, StableRNGs, LinearAlgebra, Statistics, Dates, Clarabel,
          HiGHS, Pajarito, JuMP, StatsPlots, GraphRecipes, InteractiveUtils
    po = PortfolioOptimisers
    OPS = po.OnlinePortfolioSelection

    slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                 settings = Dict("verbose" => false),
                 check_sol = (; allow_local = true, allow_almost = true))
    mip = Solver(; name = :mip,
                 solver = optimizer_with_attributes(Pajarito.Optimizer, "verbose" => false,
                                                    "oa_solver" =>
                                                        optimizer_with_attributes(HiGHS.Optimizer,
                                                                                  MOI.Silent() =>
                                                                                      true),
                                                    "conic_solver" =>
                                                        optimizer_with_attributes(Clarabel.Optimizer,
                                                                                  "verbose" =>
                                                                                      false)),
                 check_sol = (; allow_local = true, allow_almost = true))

    rng = StableRNG(7)
    T, N = 40, 4
    R = 0.02 .* randn(rng, T, N)
    nx = ["A", "B", "C", "D"]
    ts = Date(2020, 1, 1) .+ Day.(0:(T - 1))
    rd = ReturnsResult(; nx = nx, X = R, ts = ts)
    rows(r, i) = po.port_opt_view(r, i, :)
    cut(r, i) = ReturnsResult(; nx = r.nx, X = r.X[i, :], ts = r.ts[i])

    # The time-varying panel: D is unlisted over rows 1:10 and B delists from row 26.
    Rg = copy(R)
    amsk = trues(T, N)
    amsk[1:10, 4] .= false
    amsk[26:T, 2] .= false
    Rg[.!amsk] .= NaN
    rdg = ReturnsResult(; nx = nx, X = Rg, ts = ts,
                        pnl = AssetPanel(; amsk = amsk, emsk = copy(amsk)))

    # The roster: every concrete rule of the family and every constructor that changes the
    # step, each at parameters that exercise it on forty rows of four assets.
    me_ew = ExpWeightedExpectedReturns(; decay = 0.9, min_obs = 1)
    crps = [ConstantRebalancedPortfolio(; w = Float64.(1:N .== k)) for k in 1:N]
    roster = ["BuyAndHold" => BuyAndHold(),
              "ConstantRebalancedPortfolio" => ConstantRebalancedPortfolio(),
              "ConstantRebalancedPortfolio(w)" =>
                  ConstantRebalancedPortfolio(; w = [0.4, 0.3, 0.2, 0.1]),
              "NewtonStep" => NewtonStep(),
              "NewtonStep(Gram)" => NewtonStep(; proj = GramProjection(; slv = slv)),
              "PassiveAggressiveMeanReversion" => PassiveAggressiveMeanReversion(),
              "PassiveAggressiveMeanReversion(linear)" =>
                  PassiveAggressiveMeanReversion(; slack = LinearSlack()),
              "PassiveAggressiveMeanReversion(quadratic)" =>
                  PassiveAggressiveMeanReversion(; slack = QuadraticSlack()),
              "UniversalPortfolio" => UniversalPortfolio(; N = N, n_experts = 6, seed = 3),
              "ExpertMixture(EG over CRPs)" =>
                  ExpertMixture(; experts = crps, alg = ExponentiatedGradient(; eta = 0.5)),
              "ExpertMixture(BlendPoint)" =>
                  ExpertMixture(; experts = [ExponentiatedGradient(), GradientProjection()],
                                alg = ExponentiatedGradient(), grad = BlendPoint()),
              "MovingAverageReversion" => MovingAverageReversion(),
              "ExponentialMovingAverageReversion" => ExponentialMovingAverageReversion(),
              "RobustMedianReversion" => RobustMedianReversion(),
              "ReweightedPriceRelativeTracking" => ReweightedPriceRelativeTracking(),
              "GaussianWeightingReversion" => GaussianWeightingReversion(),
              "LocalAdaptiveLearning" => LocalAdaptiveLearning(),
              "PeakPriceTracking" => PeakPriceTracking(),
              "AdaptiveInputCompositeTrend" => AdaptiveInputCompositeTrend(),
              "TrendPromotePriceTracking" => TrendPromotePriceTracking(),
              "KernelTrendPatternTracking" => KernelTrendPatternTracking(),
              "TransactionCostOptimisation" => TransactionCostOptimisation(),
              "ShortTermSparsePortfolio" => ShortTermSparsePortfolio(),
              "ShortTermSparsePortfolio(L1)" =>
                  ShortTermSparsePortfolio(; alg = L1Optimum()),
              "ShortTermSparsePortfolio(ADM)" =>
                  ShortTermSparsePortfolio(; alg = AlternatingDirectionMethod()),
              "ForecastReversion(prior)" => ForecastReversion(; me = me_ew),
              "ForecastTracking(prior)" => ForecastTracking(; me = SimpleExpectedReturns()),
              "ConfidenceWeightedMeanReversion" => ConfidenceWeightedMeanReversion(),
              "ConfidenceWeightedMeanReversion(sd)" => ConfidenceWeightedMeanReversion(;
                                                                                       formulation = StandardDeviationUpdate()),
              "AntiCorrelation" => AntiCorrelation(; window = 3),
              "ExpectationMaximisation" => ExpectationMaximisation(),
              "AggregatingAlgorithm" => AggregatingAlgorithm(), "TopK" => TopK(; k = 2),
              "WeakAggregatingAlgorithm" => WeakAggregatingAlgorithm(),
              "AggregatingExponentialGradient" =>
                  AggregatingExponentialGradient(; etas = [0.05, 0.1]),
              "MirrorDescent" => MirrorDescent(),
              "ExponentiatedGradient" => ExponentiatedGradient(),
              "ExponentiatedGradient(alpha)" => ExponentiatedGradient(; alpha = 0.1),
              "GradientProjection" => GradientProjection(), "EGE" => EGE(), "EGR" => EGR(),
              "EGA" => EGA(),
              "MirrorDescent(Tsallis)" => MirrorDescent(; proj = TsallisProjection()),
              "MirrorDescent(log barrier)" =>
                  MirrorDescent(; proj = LogBarrierProjection()),
              "MirrorDescent(inverse root)" =>
                  MirrorDescent(; eta = InverseSquareRootRate(; c = 0.2)),
              "MirrorDescent(doubling)" =>
                  MirrorDescent(; eta = DoublingTrickRate(; N = 2)),
              "MirrorDescent(self-confident)" => MirrorDescent(; eta = SelfConfidentRate()),
              "MAEG" => MAEG(; etas = [0.05, 0.1], window = 3),
              "MirrorDescent(RiskLoss)" => MirrorDescent(; obj = RiskLoss(; window = 5)),
              "FollowTheLeader" => FollowTheLeader(),
              "FollowTheLeader(gamma)" => FollowTheLeader(; gamma = 0.3),
              "FollowTheLeader(LastRows)" => FollowTheLeader(; sel = LastRows(; W = 5)),
              "FollowTheLeader(HistogramMatch)" =>
                  FollowTheLeader(; sel = HistogramMatch(; window = 2)),
              "FollowTheLeader(KernelMatch)" =>
                  FollowTheLeader(; sel = KernelMatch(; window = 2, radius = 0.5)),
              "FollowTheLeader(NearestNeighbourMatch)" =>
                  FollowTheLeader(; sel = NearestNeighbourMatch(; window = 2)),
              "FollowTheLeader(CorrelationMatch)" =>
                  FollowTheLeader(; sel = CorrelationMatch(; window = 2)),
              "FollowTheLeader(ClusterMatch)" => FollowTheLeader(;
                                                                 sel = ClusterMatch(; window = 2,
                                                                                    clusterer = ClustersEstimator(;
                                                                                                                  alg = KMeansAlgorithm(;
                                                                                                                                        seed = 1)))),
              "ShortTermLossControlPortfolio" => ShortTermLossControlPortfolio(; slv = slv),
              "LowDimensionEnsemblePortfolio" =>
                  LowDimensionEnsemblePortfolio(; N = N, slv = slv,
                                                pe = LowDimensionEnsemblePrior(; seed = 1)),
              "FollowTheLeadingHistory" => FollowTheLeadingHistory(),
              "AdaptiveSubgradient" => AdaptiveSubgradient(),
              "OptimisticStep" => OptimisticStep(),
              "OptimisticStep(MeanGradient)" =>
                  OptimisticStep(; predictor = MeanGradient()),
              "OptimisticStep(ForecastGradient)" =>
                  OptimisticStep(; alg = GradientProjection(; eta = HintResidualRate()),
                                 predictor = ForecastGradient(; me = me_ew)),
              "SwitchingPortfolio" => SwitchingPortfolio(; N = N),
              "Ader" => Ader(; eta_min = 0.05, K = 3, eps = 0.3),
              "Sword" => Sword(; eta_min = 0.05, K = 3, eps = 0.3)]

    function stepped(opt, r, blocks)
        o = opt
        stop = 0
        for b in blocks
            o = po.partial_fit!(o, rows(r, (stop + 1):(stop + b)))
            stop += b
        end
        return o, stop
    end

    @testset "3. The fee-charged, drifted walk-forward and the hindsight comparators" begin
        fees = Fees(; tn = Turnover(; w = fill(1 / N, N), val = 0.001))
        cv = OnlineIndexWalkForward(20, 1; pws = DriftedWeights(),
                                    wd = SelfFinancingDrift())
        rdt = cut(rd, 21:T)
        bcrp = predict(optimise(BestConstantRebalancedPortfolio(), rdt), rdt)
        best_stock = Pipeline(;
                              steps = (ScoreSelector(; score = MeanReturn(; flag = true),
                                                     rule = RankRule(; best = 1)),
                                       EqualWeighted()))
        stock = predict(fit(best_stock, rdt), rdt)
        # The per-row Hindsight Comparator through the split shares the strategy's rows.
        hs = cross_val_predict(BestConstantRebalancedPortfolio(), rd,
                               HindsightSplit(; start = 21))
        @test hs.mrd.ts == rdt.ts
        for (name, alg) in roster
            opt = OPS(; alg = alg, fees = fees)
            pred = cross_val_predict(opt, rd, cv)
            free = cross_val_predict(OPS(; alg = alg), rd, cv)
            @test length(pred.pred) == T - 20
            @test pred.mrd.ts == rdt.ts
            @test all(isfinite, pred.mrd.X)
            # The fee is charged: net of fees never beats fee-free on the same path, and
            # the fee bites wherever the book moved.
            @test all(pred.mrd.X .<= free.mrd.X .+ 1e-14)
            @test all(p -> isa(p.res.fees, Fees), pred.pred)
            @test all(p -> !isnothing(p.hw), pred.pred)
            for comp in (bcrp, stock, hs)
                lr = log_wealth_regret(pred, comp)
                @test isa(lr, LogWealthRegretResult)
                @test isfinite(lr.regret)
                @test length(lr.difference) == T - 20
                @test isapprox(lr.regret, sum(lr.difference); atol = 1e-12)
            end
            ps = performance_summary(pred; benchmark = vec(bcrp.rd.X))
            @test isfinite(ps.excess_ret) && isfinite(ps.turnover)
        end
        # A turnover fee on the family requires the drifted book at the loop's door.
        @test_throws ArgumentError cross_val_predict(OPS(; alg = BuyAndHold(), fees = fees),
                                                     rd, OnlineIndexWalkForward(20, 1))
    end

    @testset "4. The two-regime diagnostic of the user guide reproduces" begin
        function synthetic_market(kind; T = 500, N = 5, seed = 7)
            rng = StableRNG(seed)
            P = ones(T + 1, N)
            lp = zeros(N)
            for t in 1:T
                if kind == :trending
                    lp .= 0.90 .* lp .+ 0.008 .* randn(rng, N)
                    P[t + 1, :] .= P[t, :] .* exp.(lp)
                else
                    lp .= 0.90 .* lp .+ 0.04 .* randn(rng, N)
                    P[t + 1, :] .= exp.(lp)
                end
            end
            days = filter(d -> Dates.dayofweek(d) <= 5,
                          Date(2016, 1, 4):Day(1):(Date(2016, 1, 4) + Day(2 * T + 10)))[1:T]
            return ReturnsResult(; nx = ["A$i" for i in 1:N],
                                 X = P[2:end, :] ./ P[1:(end - 1), :] .- 1, ts = days)
        end
        rd_rev = synthetic_market(:reverting)
        rd_trend = synthetic_market(:trending)
        cv = OnlineIndexWalkForward(20, 1)
        wealth(alg, r) = prod(1 .+ cross_val_predict(OPS(; alg = alg), r, cv).mrd.X)
        # The chapter's numbers: the reversion rules are a total bet on the market's
        # property, and the winner rules sit together on both markets.
        mar_rev = wealth(MovingAverageReversion(), rd_rev)
        mar_trend = wealth(MovingAverageReversion(), rd_trend)
        @test isapprox(mar_rev, 22.4; rtol = 0.01)
        @test mar_trend < 0.01
        for alg in (PassiveAggressiveMeanReversion(), RobustMedianReversion())
            @test wealth(alg, rd_rev) > 1.5
            @test wealth(alg, rd_trend) < 0.01
        end
        winners = [ConstantRebalancedPortfolio(), ExponentiatedGradient(), NewtonStep(),
                   UniversalPortfolio(; N = 5, seed = 1)]
        w_rev = [wealth(a, rd_rev) for a in winners]
        w_trend = [wealth(a, rd_trend) for a in winners]
        @test all(1.3 .< w_rev .< 1.8)
        @test all(2.6 .< w_trend .< 3.2)
        # The regret against the best constant rebalanced portfolio: negative for the
        # reversion rules on the reverting market, inside Cover's bound for the mixture.
        rdt = cut(rd_rev, 21:500)
        bcrp = predict(optimise(BestConstantRebalancedPortfolio(), rdt), rdt)
        mar = cross_val_predict(OPS(; alg = MovingAverageReversion()), rd_rev, cv)
        up = cross_val_predict(OPS(; alg = UniversalPortfolio(; N = 5, seed = 1)), rd_rev,
                               cv)
        @test log_wealth_regret(mar, bcrp).regret < -1
        @test 0 < log_wealth_regret(up, bcrp).regret < (5 - 1) * log(481)
        # The chapter's later sections, one number each. The forecast is the bet: the same
        # update fed a recent-mean forecast off a prior makes the opposite bet to the
        # moving-average one, and the two rows of its table are mirror images.
        recent = PriorExpectedReturns(;
                                      pe = EmpiricalPrior(;
                                                          me = ExpWeightedExpectedReturns(;
                                                                                          decay = 0.9,
                                                                                          min_obs = 1)))
        @test wealth(ForecastReversion(; me = recent), rd_rev) < 0.05
        @test wealth(ForecastReversion(; me = recent), rd_trend) > 500
        # The three interior geometries land within two percent of each other at one rate;
        # the Tsallis step reaches 1.465 where the entropic one reaches 1.452.
        eg_rev = wealth(ExponentiatedGradient(), rd_rev)
        ts_rev = wealth(MirrorDescent(; proj = TsallisProjection()), rd_rev)
        @test isapprox(ts_rev, 1.465; rtol = 0.01)
        @test isapprox(ts_rev, eg_rev; rtol = 0.02)
        @test isapprox(wealth(MirrorDescent(; proj = LogBarrierProjection()), rd_rev),
                       eg_rev; rtol = 0.02)
        # The leader over a matched sample is the one rule that wins on both markets.
        matched = FollowTheLeader(; sel = NearestNeighbourMatch(; window = 2))
        @test isapprox(wealth(matched, rd_rev), 2.85; rtol = 0.02)
        @test wealth(matched, rd_trend) > 1000
        # Dynamic regret: be-the-leader walks a path of 73 and the reversion rule is still
        # 0.18 ahead of it; the best stock of each row walks 577 and no rule is within
        # twenty of it.
        btl = cross_val_predict(BestConstantRebalancedPortfolio(), rdt, HindsightSplit())
        best_stock = Pipeline(;
                              steps = (ScoreSelector(; score = MeanReturn(; flag = true),
                                                     rule = RankRule(; best = 1)),
                                       EqualWeighted()))
        per_row = cross_val_predict(best_stock, rdt, HindsightSplit(; prefix = false))
        r_btl = log_wealth_regret(mar, btl)
        r_row = log_wealth_regret(mar, per_row)
        @test isapprox(r_btl.regret, -0.18; atol = 0.02)
        @test isapprox(r_btl.path_length, 73.0; rtol = 0.01)
        @test isapprox(r_row.regret, 19.9; rtol = 0.02)
        @test isapprox(r_row.path_length, 577.0; rtol = 0.01)
        crp = cross_val_predict(OPS(; alg = ConstantRebalancedPortfolio()), rd_rev, cv)
        @test isapprox(log_wealth_regret(crp, btl).regret, 2.54; rtol = 0.02)
    end
end
