@safetestset "MeanRisk Optimisation" begin
    using PortfolioOptimisers, CSV, DataFrames, Test, StableRNGs, Random, Clarabel,
          StatsBase, LinearAlgebra
    function find_tol(a1, a2; name1 = :a1, name2 = :a2)
        for rtol ∈
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; rtol = rtol)
                println("isapprox($name1, $name2, rtol = $(rtol))")
                break
            end
        end
    end
    @testset "MeanRisk single risk measure optimisation" begin
        rng = StableRNG(987456321)
        X = randn(rng, 200, 10)
        rd = ReturnsResult(; nx = 1:size(X, 2), X = X)
        pr = prior(HighOrderPriorEstimator(), rd)
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        ew = eweights(1:size(X, 1), inv(size(X, 1)); scale = true)
        w1 = fill(inv(10), 10)
        rf = 4.34 / 100 / 252
        sigma = cov(GerberCovariance(), X)
        mu = vec(mean(ShrunkExpectedReturns(; ce = GerberCovariance()), X))
        kt = cokurtosis(Cokurtosis(; alg = Semi()), X; mean = transpose(mu))
        sk, V = coskewness(Coskewness(; alg = Semi()), X; mean = transpose(mu))
        ucs1 = sigma_ucs(NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                                       rng = rng,
                                                       alg = BoxUncertaintySetAlgorithm(),
                                                       seed = 987654321), X)
        ucs2 = sigma_ucs(NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                                       rng = rng,
                                                       alg = EllipseUncertaintySetAlgorithm(),
                                                       seed = 987654321), X)
        df = CSV.read(joinpath(@__DIR__, "./assets/MeanRisk.csv"), DataFrame)
        risks = [Variance(; sigma = sigma),#
                 Variance(),#
                 Variance(; formulation = Quad()),#
                 Variance(; formulation = RSOC()),# 4
                 ###
                 UncertaintySetVariance(; sigma = sigma, ucs = ucs1),#
                 UncertaintySetVariance(; ucs = ucs2),# 6
                 ###
                 StandardDeviation(; sigma = sigma),#
                 StandardDeviation(;),# 8
                 ###
                 LowOrderMoment(; mu = mu),#
                 LowOrderMoment(; mu = rf),#
                 LowOrderMoment(; w = ew),#
                 LowOrderMoment(),# 12
                 ###
                 LowOrderMoment(; alg = SemiDeviation(), mu = mu),#
                 LowOrderMoment(; alg = SemiDeviation(), mu = rf),#
                 LowOrderMoment(; alg = SemiDeviation(), w = ew),#
                 LowOrderMoment(; alg = SemiDeviation()),# 16
                 ###
                 LowOrderMoment(; alg = SemiVariance(), mu = mu),#
                 LowOrderMoment(; alg = SemiVariance(), mu = rf),#
                 LowOrderMoment(; alg = SemiVariance(), w = ew),#
                 LowOrderMoment(; alg = SemiVariance()),# 20
                 ###
                 LowOrderMoment(; alg = MeanAbsoluteDeviation(), mu = mu),#
                 LowOrderMoment(; alg = MeanAbsoluteDeviation(), mu = rf),#
                 LowOrderMoment(; alg = MeanAbsoluteDeviation(), w = ew),#
                 LowOrderMoment(; alg = MeanAbsoluteDeviation(; w = ew)),#
                 LowOrderMoment(; alg = MeanAbsoluteDeviation(;)),# 25
                 ###
                 SquareRootKurtosis(),#
                 SquareRootKurtosis(; kt = kt),#
                 SquareRootKurtosis(; N = 2),# 28
                 ###
                 NegativeSkewness(),#
                 NegativeSkewness(; alg = QuadraticNegativeSkewness()),#
                 NegativeSkewness(; sk = sk, V = V),#
                 NegativeSkewness(; alg = QuadraticNegativeSkewness(), sk = sk, V = V),# 32
                 ###
                 ConditionalValueatRisk(),#
                 DistributionallyRobustConditionalValueatRisk(; l = 2.5, r = 0.85),#
                 ConditionalValueatRiskRange(),#
                 ConditionalDrawdownatRisk(),# 36
                 ###
                 EntropicValueatRisk(),#
                 EntropicValueatRiskRange(),#
                 EntropicDrawdownatRisk(),# 39
                 ###
                 RelativisticValueatRisk(),#
                 RelativisticValueatRiskRange(),#
                 RelativisticDrawdownatRisk(),# 42
                 ###
                 OrderedWeightsArray(),#
                 OrderedWeightsArray(; w = owa_tg(200)),#
                 OrderedWeightsArrayRange(;),# 45
                 ###
                 OrderedWeightsArray(; formulation = ExactOrderedWeightsArray()),#
                 OrderedWeightsArray(; formulation = ExactOrderedWeightsArray(),
                                     w = owa_tg(200)),#
                 OrderedWeightsArrayRange(; formulation = ExactOrderedWeightsArray()),# 48
                 ###
                 AverageDrawdown(),#
                 AverageDrawdown(; w = ew),# 50
                 ###
                 UlcerIndex(),# 51
                 ###
                 MaximumDrawdown(),# 52
                 ###
                 WorstRealisation(),# 53
                 ###
                 Range(),# 54
                 ###
                 TurnoverRiskMeasure(; w = w1),#
                 TrackingRiskMeasure(; tracking = WeightsTracking(; w = w1)),#
                 TrackingRiskMeasure(; tracking = ReturnsTracking(; w = pr.X * w1))]
        objs = [MinimumRisk(), MaximumUtility(), MaximumUtility(; l = 1),
                MaximumRatio(; ohf = 1), MaximumRatio(;), MaximumRatio(; rf = rf, ohf = 1),
                MaximumRatio(; rf = rf), MaximumReturn()]
        rets = [ArithmeticReturn(), KellyReturn()]
        i = 1
        for r ∈ risks
            for obj ∈ objs
                for ret ∈ rets
                    opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
                    sol = optimise!(MeanRiskEstimator(; r = r, obj = obj, opt = opt))
                    if isa(sol.retcode, OptimisationFailure)
                        i += 1
                        continue
                    end
                    w = sol.w
                    wt = df[!, i]
                    res = if i ∈ (92, 140, 142, 160, 364, 380, 396, 460, 474, 506, 510, 520,
                                  540, 556)
                        isapprox(w, wt; rtol = 1e-4)
                    elseif i == 726
                        isapprox(w, wt; rtol = 5e-4)
                    else
                        isapprox(w, wt; rtol = 5e-5)
                    end
                    if !res
                        println("Iteration $i failed.")
                        find_tol(w, wt; name1 = :w, name2 = :wt)
                    end
                    @test res
                    df[!, "$i"] = w
                    i += 1
                end
            end
        end
    end
    @testset "Returns lower bounds" begin
        rng = StableRNG(987456321)
        X = randn(rng, 200, 10)
        rd = ReturnsResult(; nx = 1:size(X, 2), X = X)
        pr = prior(EmpiricalPriorEstimator(), rd)
        slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                     check_sol = (; allow_local = true, allow_almost = true),
                     settings = Dict("max_step_fraction" => 0.75, "verbose" => false))
        rf = 4.34 / 100 / 252
        ucs1 = mu_ucs(NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                                    rng = rng,
                                                    alg = BoxUncertaintySetAlgorithm(),
                                                    seed = 987654321), X)
        ucs2 = mu_ucs(NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                                    rng = rng,
                                                    alg = EllipseUncertaintySetAlgorithm(),
                                                    seed = 987654321), X)

        ret = ArithmeticReturn()
        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        res1 = optimise!(mre)

        ret = ArithmeticReturn(; lb = dot(res1.w, pr.mu))
        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRiskEstimator(; obj = MinimumRisk(), opt = opt)
        res2 = optimise!(mre)
        @test abs(ret.lb - dot(res2.w, pr.mu)) <= 1e-9

        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRiskEstimator(; obj = MaximumRatio(), opt = opt)
        res3 = optimise!(mre)
        @test abs(ret.lb - dot(res3.w, pr.mu)) <= 5e-7

        ret = KellyReturn()
        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRiskEstimator(; obj = MaximumRatio(; rf = rf), opt = opt)
        res1 = optimise!(mre)

        ret = KellyReturn(; lb = mean(log1p.(pr.X * res1.w)))
        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRiskEstimator(; obj = MinimumRisk(), opt = opt)
        res2 = optimise!(mre)
        @test abs(mean(log1p.(pr.X * res2.w)) - ret.lb) <= 1e-10

        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRiskEstimator(; obj = MaximumRatio(), opt = opt)
        res3 = optimise!(mre)
        @test abs(mean(log1p.(pr.X * res3.w)) - ret.lb) <= 1e-8
    end
end
