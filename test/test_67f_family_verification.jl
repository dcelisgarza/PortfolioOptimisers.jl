#=
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

    @testset "1. The batch-online identity at every block size, and Resume" begin
        cv = IndexWalkForward(5, 1; ff = OnlineStep())
        for (name, alg) in roster
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
            # A Risk Loss over the default prior meets the unlisted leg's constant column
            # and fails in the covariance's repair, as its docstring states; over a prior
            # whose covariance skips the repair it runs.
            tv = alg
            if isa(alg, MirrorDescent) && isa(alg.obj, RiskLoss)
                @test_throws ArgumentError optimise(opt, rows(rdg, 1:30))
                tv = MirrorDescent(;
                                   obj = RiskLoss(; window = 5,
                                                  pe = EmpiricalPrior(;
                                                                      ce = PortfolioOptimisersCovariance(;
                                                                                                         mp = MatrixProcessing(;
                                                                                                                               pdm = nothing)))))
            end
            optg = OPS(; alg = tv)
            bg = optimise(optg, rows(rdg, 1:30))
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

    @testset "2. Every rule on a programme set with a turnover ceiling and a cardinality bound" begin
        # The ceiling is per asset against the Price-Adjusted Allocation; the bound is on the
        # count of non-zero legs. Both are honoured by every rule, or the step is held. The
        # start is on two legs, because no allocation on two legs lies within the ceiling of
        # the uniform one, and a set that is infeasible at the seed holds every step.
        pset = ProgrammeAllocationSet(; slv = mip, tn = 0.1, card = 2)
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
        for (name, alg) in roster
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
                @test all(res.w .>= -1e-6)
                # A Held Step trades nothing and sits inside the ceiling; a mixture whose expert
                # held still projects its blend, so the ceiling is the check in both cases.
                @test maximum(abs.(res.w .- w_adj)) <= 0.1 + 1e-5
            end
        end
    end

    @testset "3. The fee-charged, drifted walk-forward and the hindsight comparators" begin
        fees = Fees(; tn = Turnover(; w = fill(1 / N, N), val = 0.001))
        cv = IndexWalkForward(20, 1; ff = OnlineStep(), pws = DriftedWeights(),
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
                                                     rd,
                                                     IndexWalkForward(20, 1;
                                                                      ff = OnlineStep()))
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
        cv = IndexWalkForward(20, 1; ff = OnlineStep())
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
    end

    @testset "5. The prototype's corrections hold on the shipped code" begin
        # Cover's fixed point beats the best single asset: the best constant rebalanced
        # portfolio is a hindsight comparator the best stock never exceeds.
        rng2 = StableRNG(2150)
        T2 = 60
        R2 = zeros(T2, 3)
        for t in 1:T2
            s = isodd(t) ? 1 : -1
            R2[t, :] .= [0.4 * s, -0.25 * s, 0.0] .+ 0.01 .* randn(rng2, 3)
        end
        rd2 = ReturnsResult(; nx = ["A", "B", "C"], X = R2,
                            ts = Date(2021, 1, 1) .+ Day.(0:(T2 - 1)))
        bres = optimise(BestConstantRebalancedPortfolio(), rd2)
        @test isa(bres.retcode, OptimisationSuccess)
        bcrp = predict(bres, rd2)
        best_stock = Pipeline(;
                              steps = (ScoreSelector(; score = MeanReturn(; flag = true),
                                                     rule = RankRule(; best = 1)),
                                       EqualWeighted()))
        stock = predict(fit(best_stock, rd2), rd2)
        @test log_wealth_regret(stock, bcrp).regret > 0
        # Projected gradient on the same objective lands short of the fixed point: the
        # prototype's correction. One gradient step from uniform, projected, is worse.
        f(w) = sum(log.(1 .+ R2 * w))
        g = [sum((1 .+ R2 * fill(1 / 3, 3)) .\ R2[:, j]) for j in 1:3]
        one_step = po.project(po.EuclideanProjection(),
                              po.resolve_allocation_set(BoundedAllocationSet(), 3, false,
                                                        Float64),
                              fill(1 / 3, 3) .+ 0.01 .* g, fill(1 / 3, 3))
        @test f(bres.w) > f(one_step)
        @test f(bres.w) >= maximum(f(Float64.(1:3 .== k)) for k in 1:3)
        # The parity oracle is the prototype itself, kept under `research/prototypes/` and
        # loaded by the first-set test file.
        @test isfile(joinpath(@__DIR__, "..", "research", "prototypes",
                              "09_online_portfolio_selection.jl"))
    end

    @testset "6. The weight path renders, and a read-out reaches the finite allocation" begin
        cv = IndexWalkForward(20, 1; ff = OnlineStep(), pws = DriftedWeights(),
                              wd = SelfFinancingDrift(), store_weight_path = true)
        pred = cross_val_predict(OPS(; alg = PassiveAggressiveMeanReversion()), rd, cv)
        is_plot(x) = x isa Plots.Plot || x isa Plots.AbstractLayout
        @test is_plot(plot_composition(pred; N = 4))
        @test is_plot(plot_portfolio_cumulative_returns(pred; compound = true))
        @test is_plot(plot_turnover(pred))
        @test is_plot(plot_weight_stability(pred))
        # The last fold's target is a plain weight vector on the named universe; the finite
        # allocations take it as any optimiser's Result.
        w = pred.pred[end].res.w
        @test length(w) == N && isapprox(sum(w), 1; atol = 1e-12)
        prices = [100.0, 50.0, 25.0, 10.0]
        inp = FiniteAllocationInput(; w = w, prices = prices, cash = 1e6)
        ga = optimise(GreedyAllocation(), inp)
        @test isapprox(ga.w, w; atol = 1e-3)
        highs = Solver(; name = :highs, solver = HiGHS.Optimizer,
                       settings = Dict("log_to_console" => false),
                       check_sol = (; allow_local = true, allow_almost = true))
        da = optimise(DiscreteAllocation(; slv = highs), inp)
        @test isa(da.retcode, OptimisationSuccess)
        @test isapprox(da.w, w; atol = 1e-3)
        @test dot(da.shares, prices) <= 1e6
    end
end
