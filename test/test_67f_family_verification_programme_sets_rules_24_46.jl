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
              "MirrorDescent(LastGradient)" => MirrorDescent(; predictor = LastGradient()),
              "MirrorDescent(MeanGradient)" => MirrorDescent(; predictor = MeanGradient()),
              "GradientProjection(ForecastGradient)" =>
                  GradientProjection(; eta = HintResidualRate(),
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

    @testset "2. Every rule on a programme set with a turnover ceiling and a cardinality bound, rules 24 on" begin
        # The ceiling is per asset against the Price-Adjusted Allocation; the bound is on the
        # count of non-zero legs. Both are honoured by every rule, or the step is held. The
        # start is on two legs, because no allocation on two legs lies within the ceiling of
        # the uniform one, and a set that is infeasible at the seed holds every step.
        pset = ProgrammeAllocationSet(; slv = mip, tn = Turnover(; w = zeros(4), val = 0.1),
                                      card = 2)
        w0 = [0.5, 0.5, 0.0, 0.0]
        t0 = 6
        # A solved leader under a programme set runs its programme on the set's kind, so
        # its own solver is the MIP; the solver-free leader is refused by name.
        logopt = MeanRisk(; obj = MaximumReturn(),
                          opt = JuMPOptimiser(; pe = EmpiricalPrior(), slv = mip,
                                              ret = LogarithmicReturn()))
        function programme_rule(alg)
            return alg
        end
        function programme_rule(alg::FollowTheLeader{<:Any,
                                                     <:BestConstantRebalancedPortfolio})
            @test_throws ArgumentError OPS(; alg = alg, set = pset)
            return FollowTheLeader(; sel = alg.sel, opt = logopt, gamma = alg.gamma)
        end
        function programme_rule(alg::FollowTheLeader{<:Any, <:MeanRisk})
            return FollowTheLeader(; sel = alg.sel, gamma = alg.gamma,
                                   opt = MeanRisk(; obj = alg.opt.obj, r = alg.opt.r,
                                                  opt = JuMPOptimiser(; pe = alg.opt.opt.pe,
                                                                      slv = mip,
                                                                      ret = alg.opt.opt.ret,
                                                                      fees = alg.opt.opt.fees)))
        end
        for (name, alg) in roster[24:end]
            @testset "$name" begin
                opt = OPS(; alg = programme_rule(alg), set = pset, w0 = w0)
                o = po.partial_fit!(opt, rows(rd, 1:(t0 - 1)))
                w_prev = optimise(o).w
                @test count(x -> x > 1e-6, w_prev) <= 2
                @test isapprox(sum(w_prev), 1; atol = 1e-6)
                x = 1 .+ R[t0, :]
                w_adj = w_prev .* x ./ dot(w_prev, x)
                o2 = po.partial_fit!(o, rows(rd, t0:t0))
                res = optimise(o2)
                @test isa(res.retcode, OptimisationSuccess)
                @test count(x -> x > 1e-6, res.w) <= 2
                @test isapprox(sum(res.w), 1; atol = 1e-6)
                # The solver's error on a zero leg grows with the size of the projected point:
                # the sparse portfolio projects `500 b`, and its zero legs come back near
                # `-1e-6`. The sign is checked at the ceiling's tolerance.
                @test all(res.w .>= -1e-5)
                # A Held Step trades nothing and sits inside the ceiling; a mixture whose expert
                # held still projects its blend, so the ceiling is the check in both cases.
                @test maximum(abs.(res.w .- w_adj)) <= 0.1 + 1e-5
            end
        end
    end
end
