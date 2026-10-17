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
    @test allunique(first.(roster))

    # Every concrete rule type of the family is on the roster, so a type added later fails
    # here until it joins.
    concrete_rules = Set{Any}()
    walk!(t) =
        for s in subtypes(t)
            isabstracttype(s) ? walk!(s) : push!(concrete_rules, s)
        end
    walk!(po.AbstractOnlinePortfolioSelectionAlgorithm)
    rostered = Set{Any}()
    function collect_rules!(a)
        push!(rostered, typeof(a).name.wrapper)
        isa(a, ExpertMixture) && foreach(collect_rules!, a.experts)
        hasproperty(a, :alg) &&
            isa(a.alg, po.AbstractOnlinePortfolioSelectionAlgorithm) &&
            collect_rules!(a.alg)
        return nothing
    end
    foreach(collect_rules!, last.(roster))
    @test Set(Base.typename(t).wrapper for t in concrete_rules) == rostered

    function stepped(opt, r, blocks)
        o = opt
        stop = 0
        for b in blocks
            o = po.partial_fit!(o, rows(r, (stop + 1):(stop + b)))
            stop += b
        end
        return o, stop
    end

    @testset "1. The batch-online identity at every block size, and Resume, rules 1 to 15" begin
        cv = OnlineIndexWalkForward(5, 1)
        for (name, alg) in roster[1:15]
            opt = OPS(; alg = alg)
            # Static panel: four partitions of the same eighteen rows and one of all forty.
            b = optimise(opt, rows(rd, 1:18))
            @test isa(b.retcode, OptimisationSuccess)
            for blocks in ([10, 7, 1], fill(1, 18), [18], fill(3, 6), [2, 5, 11])
                o, stop = stepped(opt, rd, blocks)
                @test stop == 18
                a = optimise(o)
                @test a.w == b.w
                @test po.observation_count(o) == 18
            end
            ofull, _ = stepped(opt, rd, fill(5, 8))
            @test optimise(ofull).w == optimise(opt, rd).w
            # Time-varying panel: the same partitions over the unlisting and the delisting.
            # Every statistic over the rows reads the gap as a gap and reduces to its
            # Coverage Universe, so a Risk Loss over the default prior runs over the
            # unlisted span without meeting a constant column (ADR 0170).
            optg = opt
            bg = optimise(optg, rows(rdg, 1:30))
            @test isa(bg.retcode, OptimisationSuccess) && all(isfinite, bg.w)
            @test bg.w[2] == 0 && bg.imsk == amsk[30, :]
            for blocks in ([10, 7, 13], fill(1, 30), [30], fill(5, 6))
                og, _ = stepped(optg, rdg, blocks)
                @test optimise(og).w == bg.w
            end
            # Resume re-enters the rule where the first run stopped.
            pr = cross_val_predict(opt, rows(rd, 1:20), cv)
            pr2 = cross_val_predict(Resume(pr), rd, cv)
            one = cross_val_predict(opt, rd, cv)
            @test length(pr.pred) + length(pr2.pred) == length(one.pred)
            @test pr2.pred[end].res.w == one.pred[end].res.w
            @test po.held_timestamps(pr2.opt) == po.held_timestamps(one.opt)
            # A stepped head's view is the same recursion over the selected assets.
            v = po.port_opt_view(ofull, [1, 3])
            @test length(optimise(v).w) == 2
        end
    end
end
