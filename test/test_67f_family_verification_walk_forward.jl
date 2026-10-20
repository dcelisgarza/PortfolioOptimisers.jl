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

    @testset "3. The fee-charged, drifted walk-forward and the hindsight comparators, rules 1 to 23" begin
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
        for (name, alg) in roster[1:23]
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
    end
end
