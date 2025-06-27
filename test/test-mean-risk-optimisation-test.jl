@safetestset "MeanRisk Optimisation" begin
    using PortfolioOptimisers, CSV, DataFrames, Test, StableRNGs, Random, Clarabel,
          StatsBase, LinearAlgebra, Pajarito, HiGHS, JuMP, TimeSeries, Clustering,
          Distributions, Logging
    Logging.disable_logging(Logging.Warn)
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
        for atol ∈
            [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4,
             5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 2.5e-1, 5e-1, 1e0, 1.1e0, 1.2e0, 1.3e0,
             1.4e0, 1.5e0, 1.6e0, 1.7e0, 1.8e0, 1.9e0, 2e0, 2.5e0]
            if isapprox(a1, a2; atol = atol)
                println("isapprox($name1, $name2, atol = $(atol))")
                break
            end
        end
    end
    X = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/asset_prices.csv"));
                  timestamp = :timestamp)
    F = TimeArray(CSV.File(joinpath(@__DIR__, "./assets/factor_prices.csv"));
                  timestamp = :timestamp)
    rd = prices_to_returns(X[(end - 252):end], F[(end - 252):end])
    slv = [Solver(; name = :clarabel1, solver = Clarabel.Optimizer,
                  check_sol = (; allow_local = true, allow_almost = true),
                  settings = Dict("verbose" => false, "max_step_fraction" => 0.75))]
    mip_slv = Solver(; name = :clarabel,
                     solver = optimizer_with_attributes(Pajarito.Optimizer,
                                                        "verbose" => false,
                                                        "oa_solver" => optimizer_with_attributes(HiGHS.Optimizer,
                                                                                                 MOI.Silent() => true),
                                                        "conic_solver" => optimizer_with_attributes(Clarabel.Optimizer,
                                                                                                    "verbose" => false,
                                                                                                    "max_step_fraction" => 0.75)),
                     check_sol = (; allow_local = true, allow_almost = true))
    pr = prior(HighOrderPriorEstimator(;
                                       pe = FactorPriorEstimator(;
                                                                 re = DimensionReductionRegression())),
               rd)
    rf = 4.34 / 100 / 252
    clr = clusterise(ClusteringEstimator(), pr.X)
    clusters = cutree(clr.clustering; k = clr.k)
    sets = DataFrame(; Assets = colnames(X), Clusters = clusters)
    T, N = size(pr.X)
    eqw = range(; start = inv(N), stop = inv(N), length = N)
    @testset "MeanRisk measures" begin
        T, N = size(pr.X)
        ew = eweights(1:T, inv(T); scale = true)
        w1 = fill(inv(N), N)
        sigma = cov(GerberCovariance(), pr.X)
        mu = vec(mean(ShrunkExpectedReturns(; ce = GerberCovariance()), pr.X))
        sk, V = coskewness(Coskewness(;), rd.X)
        kt = cokurtosis(Cokurtosis(;), rd.X)
        pr.sk .= sk
        pr.V .= V
        pr.kt .= kt
        sk, V = coskewness(Coskewness(; alg = Semi()), rd.X)
        kt = cokurtosis(Cokurtosis(; alg = Semi()), rd.X)
        rng = StableRNG(987456321)
        ucs1 = sigma_ucs(NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                                       rng = rng,
                                                       alg = BoxUncertaintySetAlgorithm(),
                                                       seed = 987654321), pr.X)
        ucs2 = sigma_ucs(NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                                       rng = rng,
                                                       alg = EllipseUncertaintySetAlgorithm(),
                                                       seed = 987654321), pr.X)
        df = CSV.read(joinpath(@__DIR__, "./assets/MeanRisk1.csv"), DataFrame)
        pw = pweights(collect(range(; start = inv(size(pr.X, 1)), stop = inv(size(pr.X, 1)),
                                    length = size(pr.X, 1))))
        rs = [Variance(; sigma = sigma, formulation = QuadRiskExpr()), Variance(),
              UncertaintySetVariance(; sigma = sigma, ucs = ucs1),
              UncertaintySetVariance(; ucs = ucs2), StandardDeviation(; sigma = sigma),
              StandardDeviation(), LowOrderMoment(; mu = mu), LowOrderMoment(; mu = rf),
              LowOrderMoment(; w = pw), LowOrderMoment(),
              LowOrderMoment(;
                             alg = LowOrderDeviation(;
                                                     alg = SecondLowerMoment(;
                                                                             formulation = SqrtRiskExpr())),
                             mu = mu),
              LowOrderMoment(;
                             alg = LowOrderDeviation(;
                                                     alg = SecondLowerMoment(;
                                                                             formulation = SqrtRiskExpr())),
                             mu = rf),
              LowOrderMoment(;
                             alg = LowOrderDeviation(;
                                                     ve = SimpleVariance(;
                                                                         corrected = false,
                                                                         w = pw),
                                                     alg = SecondLowerMoment(;
                                                                             formulation = SqrtRiskExpr())),
                             w = pw),
              LowOrderMoment(;
                             alg = LowOrderDeviation(;
                                                     alg = SecondLowerMoment(;
                                                                             formulation = SqrtRiskExpr()))),
              LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment()),
                             mu = mu),
              LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment()),
                             mu = rf),
              LowOrderMoment(;
                             alg = LowOrderDeviation(;
                                                     ve = SimpleVariance(;
                                                                         corrected = false,
                                                                         w = pw),
                                                     alg = SecondLowerMoment()), w = pw),
              LowOrderMoment(; alg = LowOrderDeviation(; alg = SecondLowerMoment())),
              LowOrderMoment(;
                             alg = LowOrderDeviation(;
                                                     alg = SecondLowerMoment(;
                                                                             formulation = QuadRiskExpr())),
                             mu = mu),
              LowOrderMoment(;
                             alg = LowOrderDeviation(;
                                                     alg = SecondLowerMoment(;
                                                                             formulation = QuadRiskExpr())),
                             mu = rf),
              LowOrderMoment(;
                             alg = LowOrderDeviation(;
                                                     ve = SimpleVariance(;
                                                                         corrected = false,
                                                                         w = pw),
                                                     alg = SecondLowerMoment(;
                                                                             formulation = QuadRiskExpr())),
                             w = pw),
              LowOrderMoment(;
                             alg = LowOrderDeviation(;
                                                     alg = SecondLowerMoment(;
                                                                             formulation = QuadRiskExpr()))),
              LowOrderMoment(;
                             alg = LowOrderDeviation(;
                                                     alg = SecondLowerMoment(;
                                                                             formulation = RSOCRiskExpr())),
                             mu = mu),
              LowOrderMoment(;
                             alg = LowOrderDeviation(;
                                                     alg = SecondLowerMoment(;
                                                                             formulation = RSOCRiskExpr())),
                             mu = rf),
              LowOrderMoment(;
                             alg = LowOrderDeviation(;
                                                     ve = SimpleVariance(;
                                                                         corrected = false,
                                                                         w = pw),
                                                     alg = SecondLowerMoment(;
                                                                             formulation = RSOCRiskExpr())),
                             w = pw),
              LowOrderMoment(;
                             alg = LowOrderDeviation(;
                                                     alg = SecondLowerMoment(;
                                                                             formulation = RSOCRiskExpr()))),
              LowOrderMoment(; alg = MeanAbsoluteDeviation(), mu = mu),
              LowOrderMoment(; alg = MeanAbsoluteDeviation(), mu = rf),
              LowOrderMoment(; alg = MeanAbsoluteDeviation(), w = ew),
              LowOrderMoment(; alg = MeanAbsoluteDeviation()), SquareRootKurtosis(),
              SquareRootKurtosis(; kt = kt), SquareRootKurtosis(; N = 2),
              SquareRootKurtosis(; kt = kt, N = 2), NegativeSkewness(),
              NegativeSkewness(; alg = QuadRiskExpr()), NegativeSkewness(; sk = sk, V = V),
              NegativeSkewness(; alg = QuadRiskExpr(), sk = sk, V = V),
              ValueatRisk(; formulation = DistributionValueatRisk()),
              ValueatRisk(; formulation = DistributionValueatRisk(; dist = TDist(30))),
              ValueatRisk(; formulation = DistributionValueatRisk(; dist = Laplace())),
              ValueatRiskRange(; beta = 0.2, formulation = DistributionValueatRisk()),
              ValueatRiskRange(; beta = 0.2,
                               formulation = DistributionValueatRisk(; dist = TDist(30))),
              ValueatRiskRange(; beta = 0.2,
                               formulation = DistributionValueatRisk(; dist = Laplace())),
              ConditionalValueatRisk(;), ConditionalValueatRisk(; w = pw),
              DistributionallyRobustConditionalValueatRisk(; l = 1.7, r = 3e-4),
              DistributionallyRobustConditionalValueatRisk(; l = 1.7, r = 3e-4, w = pw),
              ConditionalValueatRiskRange(), ConditionalValueatRiskRange(; w = pw),
              DistributionallyRobustConditionalValueatRiskRange(; l_a = 1.7, r_a = 3e-4,
                                                                l_b = 1.7, r_b = 3e-4),
              DistributionallyRobustConditionalValueatRiskRange(; l_a = 1.7, r_a = 3e-4,
                                                                l_b = 1.7, r_b = 3e-4,
                                                                w = pw),
              ConditionalDrawdownatRisk(), EntropicValueatRisk(),
              EntropicValueatRisk(; w = pw), EntropicValueatRiskRange(),
              EntropicValueatRiskRange(; w = pw), EntropicDrawdownatRisk(),
              RelativisticValueatRisk(), RelativisticValueatRisk(; w = pw),
              RelativisticValueatRiskRange(), RelativisticValueatRiskRange(; w = pw),
              RelativisticDrawdownatRisk(), AverageDrawdown(), AverageDrawdown(; w = pw),
              UlcerIndex(), MaximumDrawdown(), WorstRealisation(), Range(),
              [ConditionalValueatRisk(), TurnoverRiskMeasure(; w = w1)],
              [ConditionalValueatRisk(; settings = RiskMeasureSettings(; scale = 0.5)),
               TrackingRiskMeasure(; tracking = WeightsTracking(; w = w1))],
              [ConditionalValueatRisk(; settings = RiskMeasureSettings(; scale = 0.5)),
               TrackingRiskMeasure(; tracking = ReturnsTracking(; w = pr.X * w1))]]
        objs = [MinimumRisk(), MaximumUtility(), MaximumRatio(; ohf = 1, rf = rf),
                MaximumReturn()]
        rets = [ArithmeticReturn(), KellyReturn()]
        i = 1
        for r ∈ rs
            for obj ∈ objs
                for ret ∈ rets
                    opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
                    sol = optimise!(MeanRisk(; r = r, obj = obj, opt = opt))
                    if isa(sol.retcode, OptimisationFailure) || Sys.isapple() && i == 569
                        continue
                    end
                    w = sol.w
                    wt = df[!, "$i"]
                    rtol = if i ∈
                              (6, 10, 14, 17, 18, 19, 20, 21, 22, 30, 46, 62, 70, 78, 158,
                               179, 187, 189, 213, 221, 229, 237, 245, 253, 261, 267, 275,
                               285, 291, 293, 301, 309, 317, 325, 333, 341, 349, 357, 365,
                               368, 370, 371, 374, 376, 377, 385, 393, 396, 399, 400, 403,
                               407, 415, 418, 419, 420, 423, 426, 427, 428, 430, 431, 434,
                               435, 436, 437, 438, 441, 442, 443, 444, 446, 449, 450, 451,
                               457, 462, 469, 470, 486, 518, 526, 534, 542, 556, 558, 564,
                               566) ||
                              Sys.isapple() &&
                              i ∈ (38, 54, 60, 140, 156, 171, 173, 203, 259, 269)
                        1
                    else
                        1e-6
                    end
                    res = isapprox(w, wt; rtol = rtol)
                    if !res
                        println("$i failed:\n$(typeof(r))\n$(typeof(obj))\n$(typeof(ret)).")
                        find_tol(w1, wt; name1 = :w1, name2 = :wt)
                        display([w wt])
                    end
                    @test res
                    i += 1
                end
            end
        end
    end
    @testset "Scalarisers" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv)
        r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        mr = MeanRisk(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w1 = optimise!(mr, rd).w
        @test isapprox(w1,
                       [4.0186333662477883e-10, 7.310913910954543e-10,
                        2.3266421187040205e-10, 1.8753996140279747e-9,
                        5.142392893106318e-10, 1.0598044395674972e-9, 3.3972351902684516e-9,
                        5.680737442966312e-10, 3.4851374733122416e-9, 2.141094930603568e-9,
                        0.18893964347098194, 1.2942176238414194e-9, 6.355029757289089e-10,
                        2.6757019401968575e-9, 7.282526522028215e-10, 7.963341038715699e-10,
                        4.769772925105858e-9, 2.9444193143035667e-10, 1.6584566526379706e-9,
                        0.3467123103518232, 3.9665636191862785e-8, 1.3030364138233876e-8,
                        1.0123382757395178e-9, 3.3489997015170443e-10,
                        1.0181026388913265e-9, 0.4415069625926968, 7.749835735393912e-8,
                        0.022840922470920157, 2.549363478309274e-10, 1.0396585245433163e-9],
                       rtol = 1e-6)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sce = MaxScalariser())
        r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        mr = MeanRisk(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w2 = optimise!(mr, rd).w

        opt = JuMPOptimiser(; pe = pr, slv = slv, sce = LogSumExpScalariser(; gamma = 1e-2))
        r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        mr = MeanRisk(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w3 = optimise!(mr, rd).w
        @test isapprox(w3, w1, rtol = 5e-4)

        opt = JuMPOptimiser(; pe = pr, slv = slv, sce = LogSumExpScalariser(; gamma = 1e5))
        r = [StandardDeviation(), LowOrderMoment(; alg = MeanAbsoluteDeviation())]
        mr = MeanRisk(; r = r, obj = MaximumRatio(; rf = rf), opt = opt)
        w4 = optimise!(mr, rd).w
        @test isapprox(w4, w2, rtol = 5e-5)
    end
    @testset "Returns lower bounds and uncertainty sets" begin
        rng = StableRNG(123456789)
        ucs1 = mu_ucs(NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                                    rng = rng,
                                                    alg = BoxUncertaintySetAlgorithm(),
                                                    seed = 987654321), pr.X)
        ucs2 = mu_ucs(NormalUncertaintySetEstimator(; pe = EmpiricalPriorEstimator(),
                                                    rng = rng,
                                                    alg = EllipseUncertaintySetAlgorithm(),
                                                    seed = 987654321), pr.X)

        ret = ArithmeticReturn()
        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res1 = optimise!(mre)

        ret = ArithmeticReturn(; lb = dot(res1.w, pr.mu))
        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        res2 = optimise!(mre)
        @test dot(res2.w, pr.mu) >= ret.lb - sqrt(eps())

        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRisk(; obj = MaximumRatio(), opt = opt)
        res3 = optimise!(mre)
        @test dot(res3.w, pr.mu) >= ret.lb - sqrt(eps())

        ret = KellyReturn()
        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res1 = optimise!(mre)

        ret = KellyReturn(; lb = mean(log1p.(pr.X * res1.w)))
        opt = JuMPOptimiser(; pe = pr, ret = ret,
                            slv = Solver(; name = :clarabel, solver = Clarabel.Optimizer,
                                         check_sol = (; allow_local = true,
                                                      allow_almost = true),
                                         settings = Dict("max_step_fraction" => 0.65,
                                                         "verbose" => false)))
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        res2 = optimise!(mre)
        @test mean(log1p.(pr.X * res2.w)) >= ret.lb - sqrt(eps())

        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRisk(; obj = MaximumRatio(), opt = opt)
        res3 = optimise!(mre)
        @test mean(log1p.(pr.X * res3.w)) >= ret.lb - sqrt(eps())

        ew = eweights(1:size(pr.X, 1), inv(size(pr.X, 1)); scale = true)
        ret = KellyReturn(; w = ew)
        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res1 = optimise!(mre)

        ret = KellyReturn(; lb = mean(log1p.(pr.X * res1.w), ew))
        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        res2 = optimise!(mre)
        @test mean(log1p.(pr.X * res2.w)) >= ret.lb - sqrt(eps())

        opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res3 = optimise!(mre)
        @test mean(log1p.(pr.X * res3.w)) >= ret.lb - sqrt(eps())

        ucss = (ucs1, ucs2)
        objs = (MinimumRisk(), MaximumRatio(; rf = rf))
        df = CSV.read(joinpath(@__DIR__, "./assets/MeanRisk-UncertaintyReturns.csv"),
                      DataFrame)
        i = 1
        for ucs ∈ ucss
            for obj ∈ objs
                ret = ArithmeticReturn(; ucs = ucs)
                opt = JuMPOptimiser(; pe = pr, ret = ret, slv = slv)
                mre = MeanRisk(; obj = obj, opt = opt)
                res1 = optimise!(mre)
                w = res1.w
                wt = df[!, i]
                rtol = 1e-6
                res = isapprox(w, wt; rtol = rtol)
                if !res
                    println("Iteration $i failed.")
                    find_tol(w, wt; name1 = :w, name2 = :wt)
                    display([w wt])
                end
                @test res
                i += 1
            end
        end
    end
    @testset "Budget" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv, bgt = BudgetRange(; lb = 0.8, ub = 1.5))
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        w = optimise!(mre, rd).w
        @test 0.8 <= sum(w) <= 1.5

        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test 0.8 <= sum(w) <= 1.5

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.1, ub = 0.15), bgt = 1.3,
                            sbgt = nothing)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test all(-0.1 - sqrt(eps()) .<= w .<= 0.15 + sqrt(eps()))
        @test isapprox(sum(w), 1.3)
        @test isapprox(sum(w[w .< 0]), 0)

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.1, ub = 0.15), bgt = nothing,
                            sbgt = 0.4)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test all(-0.1 - sqrt(eps()) .<= w .<= 0.15 + sqrt(eps()))
        @test isapprox(sum(w[w .< 0]), -0.4, rtol = 5e-5)

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.1, ub = 0.15), bgt = 1.3,
                            sbgt = 0.3)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test all(-0.1 - sqrt(eps()) .<= w .<= 0.15 + sqrt(eps()))
        @test isapprox(sum(w), 1.3)
        @test isapprox(sum(w[w .< 0]), -0.3, rtol = 5e-5)

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.1, ub = 0.15),
                            bgt = BudgetRange(; lb = 0.6, ub = 0.8), sbgt = 0.25)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test all(-0.1 - sqrt(eps()) .<= w .<= 0.15 + sqrt(eps()))
        @test 0.6 - sqrt(eps()) < sum(w) < 0.8 + sqrt(eps())
        @test 0.6 + 0.25 - sqrt(eps()) <= sum(w[w .> 0]) <= 0.8 + 0.25 + sqrt(eps())

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.1, ub = 0.15),
                            sbgt = BudgetRange(; lb = 0.2, ub = 0.6), bgt = 0.9)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test all(-0.1 - sqrt(eps()) .<= w .<= 0.15 + sqrt(eps()))
        @test isapprox(sum(w), 0.9)
        @test -0.6 - sqrt(eps()) <= sum(w[w .< 0]) <= -0.2 + sqrt(eps())

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.1, ub = 0.15),
                            bgt = BudgetRange(; lb = 0.6, ub = 0.8), sbgt = nothing)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test all(-0.1 - sqrt(eps()) .<= w .<= 0.15 + sqrt(eps()))
        @test 0.6 - sqrt(eps()) < sum(w) < 0.8 + sqrt(eps())

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.1, ub = 0.15),
                            sbgt = BudgetRange(; lb = 0.2, ub = 0.6), bgt = nothing)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test all(-0.1 - sqrt(eps()) .<= w .<= 0.15 + sqrt(eps()))
        @test -0.6 - sqrt(eps()) <= sum(w[w .< 0]) <= -0.2 + sqrt(eps())

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.13, ub = 0.25),
                            bgt = BudgetRange(; lb = 0.6, ub = 0.8),
                            sbgt = BudgetRange(; lb = 0.2, ub = 0.5))
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test all(-0.13 - sqrt(eps()) .<= w .<= 0.25 + sqrt(eps()))
        @test 0.6 - sqrt(eps()) < sum(w) < 0.8 + sqrt(eps())
        @test 0.2 + 0.6 - sqrt(eps()) <= sum(w[w .>= 0]) <= 0.8 + 0.5 + sqrt(eps())
        @test -0.5 - sqrt(eps()) <= sum(w[w .< 0]) <= -0.2 + sqrt(eps())

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.13, ub = 0.25),
                            bgt = BudgetRange(; lb = 0.6, ub = 0.8),
                            sbgt = BudgetRange(; lb = 0.2, ub = nothing))
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test all(-0.13 - sqrt(eps()) .<= w .<= 0.25 + sqrt(eps()))
        @test 0.6 - sqrt(eps()) < sum(w) < 0.8 + sqrt(eps())
        @test (0.2 + 0.6 - 375 * sqrt(eps())) <= sum(w[w .>= 0]) <= 0.8 + sqrt(eps())
        @test isapprox(sum(w[w .< 0]), -0.2, rtol = 5e-5)

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.13, ub = 0.25),
                            bgt = BudgetRange(; lb = 0.6, ub = 0.8),
                            sbgt = BudgetRange(; lb = nothing, ub = 0.5))
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test all(-0.13 - sqrt(eps()) .<= w .<= 0.25 + sqrt(eps()))
        @test 0.6 - sqrt(eps()) < sum(w) < 0.8 + sqrt(eps())
        @test 0.2 + 0.6 - sqrt(eps()) <= sum(w[w .>= 0]) <= 0.8 + 0.5 + sqrt(eps())
        @test -0.5 - sqrt(eps()) <= sum(w[w .< 0])

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.13, ub = 0.25),
                            bgt = BudgetRange(; lb = nothing, ub = 0.8),
                            sbgt = BudgetRange(; lb = 0.2, ub = 0.5))
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test all(-0.13 - sqrt(eps()) .<= w .<= 0.25 + sqrt(eps()))
        @test sum(w) < 0.8 + sqrt(eps())
        @test sum(w[w .>= 0]) <= 0.8 + 0.5 + sqrt(eps())
        @test -0.5 - sqrt(eps()) <= sum(w[w .< 0]) <= -0.2 + sqrt(eps())

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.13, ub = 0.25),
                            bgt = BudgetRange(; lb = 0.6, ub = nothing),
                            sbgt = BudgetRange(; lb = 0.2, ub = 0.5))
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test all(-0.13 - sqrt(eps()) .<= w .<= 0.25 + sqrt(eps()))
        @test 0.6 - sqrt(eps()) < sum(w)
        @test 0.2 + 0.6 - sqrt(eps()) <= sum(w[w .>= 0])
        @test -0.5 - sqrt(eps()) <= sum(w[w .< 0]) <= -0.2 + sqrt(eps())
    end
    @testset "Cardinality" begin
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, card = 5)
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test round(inv(dot(w, w))) <= 5

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                            sbgt = 0.3, card = 7)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        lw = w[w .>= 0]
        sw = w[w .< 0]
        @test round(1.3^2 / dot(lw, lw)) + round(0.3^2 / dot(sw, sw)) <= 7

        gcard = CardinalityConstraint(;
                                      A = CardinalityConstraintSide(; group = :Clusters,
                                                                    name = 1), B = 5,
                                      comp = LEQ())
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                            sbgt = 0.2, gcard = gcard, sets = sets)
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        swc = w[sets[!, :Clusters] .== 1]
        @test round(sum(swc) * sum(swc) / dot(swc, swc)) <= 5

        gcard = linear_constraints(gcard, sets)
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                            sbgt = 0.2, gcard = gcard)
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        @test isapprox(w, optimise!(mre, rd).w)

        gcard = CardinalityConstraint(;
                                      A = CardinalityConstraintSide(;
                                                                    group = [:Clusters,
                                                                             :Clusters,
                                                                             :Clusters],
                                                                    name = [1, 2, 3],
                                                                    coef = [1, 1, 1]),
                                      B = 2, comp = LEQ())
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                            sbgt = 0.2, gcard = gcard, sets = sets)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test sum([abs(sum(w[sets[!, :Clusters] .== i])) >= sqrt(eps()) for i ∈ 1:3]) <= 2

        gnames = [:T, :GOOG, :GILD, :MSFT, :NFLX]
        gcard = CardinalityConstraint(;
                                      A = CardinalityConstraintSide(;
                                                                    group = [:Assets,
                                                                             :Assets,
                                                                             :Assets,
                                                                             :Assets,
                                                                             :Assets],
                                                                    name = gnames,
                                                                    coef = [1, 1, 1, 1, 1]),
                                      B = 1, comp = LEQ())
        gcard = linear_constraints(gcard, sets)
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, gcard = gcard, sets = sets)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        idx = [findfirst(x -> x == string(n), rd.nx) for n ∈ gnames]
        @test sum(w[idx] .>= eps()) == 1

        gnames = [:T, :GOOG, :GILD, :MSFT, :NFLX, :TSLA]
        gcard = CardinalityConstraint(;
                                      A = CardinalityConstraintSide(;
                                                                    group = [:Assets,
                                                                             :Assets,
                                                                             :Assets,
                                                                             :Assets,
                                                                             :Assets,
                                                                             :Assets],
                                                                    name = gnames,
                                                                    coef = [1, 1, 1, 1, 1,
                                                                            1]), B = 1,
                                      comp = LEQ())
        gcard = linear_constraints(gcard, sets)
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, gcard = gcard, sets = sets)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf),
                       opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        idx = [findfirst(x -> x == string(n), rd.nx) for n ∈ gnames]
        @test sum(w[idx] .>= sqrt(eps())) == 1

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, scard = 1, smtx = :Clusters,
                            sets = sets)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test isapprox([sum(w[res.smtx[i, :]]) for i ∈ axes(res.smtx, 1)], [0, 0, 1])

        sgcard = LinearConstraint(;
                                  A = LinearConstraintSide(; group = [:Clusters, :Clusters],
                                                           name = [1, 3],
                                                           coef = [inv(6), inv(11)]), B = 1,
                                  comp = LEQ())
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sgcard = sgcard, smtx = :Clusters,
                            sets = sets)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test sum([sum(w[res.smtx[i, :]]) >= sqrt(eps()) for i ∈ [1, 3]]) <= 1

        sgcard = LinearConstraint(;
                                  A = LinearConstraintSide(; group = [:Clusters, :Clusters],
                                                           name = [1, 2],
                                                           coef = [inv(6), inv(13)]), B = 1,
                                  comp = LEQ())
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sgcard = sgcard, smtx = :Clusters,
                            sets = sets, wb = WeightBoundsResult(; lb = -0.2, ub = 1),
                            bgt = 1, sbgt = 0.2)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test sum([sum(w[res.smtx[i, :]]) >= sqrt(eps()) for i ∈ [1, 2]]) <= 1

        sgcard = LinearConstraint(;
                                  A = LinearConstraintSide(; group = [:Clusters, :Clusters],
                                                           name = [1, 3],
                                                           coef = [inv(6), inv(11)]), B = 1,
                                  comp = LEQ())
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sgcard = sgcard, smtx = :Clusters,
                            sets = sets)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf),
                       opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test sum([sum(w[res.smtx[i, :]]) >= sqrt(eps()) for i ∈ [1, 3]]) <= 1

        sgcard = LinearConstraint(;
                                  A = LinearConstraintSide(; group = [:Clusters, :Clusters],
                                                           name = [2, 3],
                                                           coef = [inv(11), inv(13)]),
                                  B = 1, comp = LEQ())
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, sgcard = sgcard, smtx = :Clusters,
                            sets = sets, wb = WeightBoundsResult(; lb = -0.2, ub = 1),
                            bgt = 1, sbgt = 0.2)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf),
                       opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test sum([sum(w[res.smtx[i, :]]) >= sqrt(eps()) for i ∈ [2, 3]]) <= 1

        plc = IntegerPhilogenyConstraintEstimator(; pe = NetworkEstimator(), B = 1)
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, bgt = 1, cplg = plc,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), sbgt = 0.2)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf),
                       opt = opt)
        res = optimise!(mre, rd)
        w = res.w

        res.cplg.A
        @test all(res.cplg.A * value.(res.model[:ib]) .<= 1)

        plc = IntegerPhilogenyConstraintEstimator(; pe = ClusteringEstimator(), B = 2)
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, bgt = 1, nplg = plc,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), sbgt = 0.2)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf),
                       opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test all(res.nplg.A * value.(res.model[:ib]) .<= 2)

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, lt = 0.2)
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test all(w[w .>= sqrt(eps())] .>= 0.2 - sqrt(eps()))

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, lt = 0.5)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test all(w[w .>= sqrt(eps())] .>= 0.5 - sqrt(eps()))

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, lt = 0.2, st = 0.2, sbgt = 1, bgt = 1,
                            wb = WeightBoundsResult(; lb = -1, ub = 1))
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test all(w[w .>= sqrt(eps())] .>= 0.2 - sqrt(eps()))
        @test all(w[w .<= -sqrt(eps())] .<= -0.2 + sqrt(eps()))

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, lt = 0.5, st = 0.5, sbgt = 1, bgt = 1,
                            wb = WeightBoundsResult(; lb = -1, ub = 1))
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test all(w[w .>= sqrt(eps())] .>= 0.5 - sqrt(eps()))
        @test all(w[w .<= -sqrt(eps())] .<= -0.5 + sqrt(eps()))
    end
    @testset "L1 and L2 penalties" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        w0 = optimise!(mre, rd).w

        opt = JuMPOptimiser(; pe = pr, slv = slv, l1 = 5e-6,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        w1 = optimise!(mre, rd).w

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, l1 = 1e-4,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1)
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        w2 = optimise!(mre, rd).w
        @test sum(w0[w0 .< 0]) <= sum(w1[w1 .< 0]) <= sum(w2[w2 .< 0])

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w3 = optimise!(mre, rd).w

        opt = JuMPOptimiser(; pe = pr, slv = slv, l1 = 1e-4,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w4 = optimise!(mre, rd).w

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, l1 = 1e-3,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w5 = optimise!(mre, rd).w
        @test sum(w3[w3 .< 0]) <= sum(w4[w4 .< 0]) <= sum(w5[w5 .< 0])

        N = size(pr.X, 2)
        wt = eqw
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, l2 = 1e-4,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1)
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        w6 = optimise!(mre, rd).w

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, l2 = 1e-3,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1)
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        w7 = optimise!(mre, rd).w
        @test rmsd(w0, wt) >= rmsd(w6, wt) >= rmsd(w7, wt)

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, l2 = 1e-2,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w8 = optimise!(mre, rd).w

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, l2 = 1e-2,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w9 = optimise!(mre, rd).w
        @test rmsd(w3, wt) >= rmsd(w8, wt) >= rmsd(w9, wt)
    end
    @testset "Fees" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv, fees = Fees(; l = 10),
                            wb = WeightBoundsResult(; lb = 0, ub = 1), sbgt = 0, bgt = 1)
        mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        @test count(res.w .>= 5e-5) == 1
        @test isapprox(sum(res.w[res.w .>= 5e-5]), 1, rtol = 5.0e-5)

        opt = JuMPOptimiser(; pe = pr, slv = slv, fees = Fees(; s = 10),
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), sbgt = 0.6,
                            bgt = 1)
        mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        @test count(res.w .<= -5e-5) == 3
        @test isapprox(sum(res.w[res.w .<= -5e-5]), -0.6, rtol = 5e-4)

        opt = JuMPOptimiser(; pe = pr, slv = slv, fees = Fees(; l = 0.0034375),
                            wb = WeightBoundsResult(; lb = 0, ub = 1), sbgt = 0, bgt = 1)
        mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MaximumRatio(; rf = rf),
                       opt = opt)
        res = optimise!(mre, rd)
        @test count(res.w .>= sqrt(eps())) == 1
        @test isapprox(sum(res.w[res.w .>= sqrt(eps())]), 1)

        opt = JuMPOptimiser(; pe = pr, slv = slv, fees = Fees(; s = 0.008056640625),
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
        mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MaximumRatio(; rf = rf),
                       opt = opt)
        res = optimise!(mre, rd)
        @test count(res.w .<= -sqrt(eps())) == 1
        @test isapprox(sum(res.w[res.w .<= -sqrt(eps())]), -1)

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, fees = Fees(; fl = 10),
                            wb = WeightBoundsResult(; lb = 0, ub = 1), sbgt = 0, bgt = 1)
        mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        @test count(res.w .>= sqrt(eps())) == 1
        @test isapprox(sum(res.w[res.w .>= sqrt(eps())]), 1)

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, fees = Fees(; fs = 10),
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
        mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        @test count(res.w .<= -sqrt(eps())) == 0
        @test isapprox(sum(res.w[res.w .<= -sqrt(eps())]), 0)

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, fees = Fees(; fl = 10),
                            wb = WeightBoundsResult(; lb = 0, ub = 1), sbgt = 0, bgt = 1)
        mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MaximumRatio(; rf = rf),
                       opt = opt)
        res = optimise!(mre, rd)
        @test count(res.w .>= sqrt(eps())) == 1
        @test isapprox(sum(res.w[res.w .>= sqrt(eps())]), 1)

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, fees = Fees(; fs = 10),
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
        mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MaximumRatio(; rf = rf),
                       opt = opt)
        res = optimise!(mre, rd)
        @test count(res.w .<= -sqrt(eps())) == 0
        @test isapprox(sum(res.w[res.w .<= -sqrt(eps())]), 0)

        opt = JuMPOptimiser(; pe = pr, slv = slv, fees = Fees(; l = 10),
                            ret = KellyReturn(), wb = WeightBoundsResult(; lb = 0, ub = 1),
                            sbgt = 0, bgt = 1)
        mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        @test count(res.w .>= 5e-5) == 1
        @test isapprox(sum(res.w[res.w .>= 5e-5]), 1, rtol = 5e-5)

        opt = JuMPOptimiser(; pe = pr, slv = slv, fees = Fees(; s = 10),
                            ret = KellyReturn(),
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), sbgt = 0.6,
                            bgt = 1)
        mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        @test count(res.w .<= -5e-5) == 3
        @test isapprox(sum(res.w[res.w .<= -5e-5]), -0.6, rtol = 5e-4)

        opt = JuMPOptimiser(; pe = pr, slv = slv, fees = Fees(; l = 10),
                            ret = KellyReturn(), wb = WeightBoundsResult(; lb = 0, ub = 1),
                            sbgt = 0, bgt = 1)
        mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MaximumRatio(; rf = rf),
                       opt = opt)
        res = optimise!(mre, rd)
        @test rmsd(res.w, eqw) <= 0.015

        opt = JuMPOptimiser(; pe = pr, slv = slv, fees = Fees(; s = 10),
                            ret = KellyReturn(), wb = WeightBoundsResult(; lb = -1, ub = 1),
                            sbgt = 1, bgt = 1)
        mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MaximumRatio(; rf = rf),
                       opt = opt)
        res = optimise!(mre, rd)
        @test rmsd(res.w, eqw) <= 0.017

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, fees = Fees(; fl = 10),
                            ret = KellyReturn(), wb = WeightBoundsResult(; lb = 0, ub = 1),
                            sbgt = 0, bgt = 1)
        mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        @test count(res.w .>= 5e-5) == 1
        @test isapprox(sum(res.w[res.w .>= 5e-5]), 1, rtol = 5e-5)

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, fees = Fees(; fs = 10),
                            ret = KellyReturn(),
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), sbgt = 0.6,
                            bgt = 1)
        mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MinimumRisk(), opt = opt)
        res = optimise!(mre, rd)
        @test count(res.w .<= -5e-5) == 0
        @test isapprox(sum(res.w[res.w .<= -5e-5]), 0, rtol = 5e-4)

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, fees = Fees(; fl = 10),
                            ret = KellyReturn(), wb = WeightBoundsResult(; lb = 0, ub = 1),
                            sbgt = 0, bgt = 1)
        mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MaximumRatio(; rf = rf),
                       opt = opt)
        res = optimise!(mre, rd)
        @test rmsd(res.w, eqw) <= 0.46

        opt = JuMPOptimiser(; pe = pr, slv = mip_slv, fees = Fees(; fs = 10),
                            ret = KellyReturn(), wb = WeightBoundsResult(; lb = -1, ub = 1),
                            sbgt = 1, bgt = 1)
        mre = MeanRisk(; r = ConditionalDrawdownatRisk(), obj = MaximumRatio(; rf = rf),
                       opt = opt)
        res = optimise!(mre, rd)
        @test count(res.w .<= -5e-5) == 0
        @test isapprox(sum(res.w[res.w .<= -5e-5]), 0, rtol = 5e-4)
    end
    @testset "Cone constraints" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv, nea = 7.5,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
        mre = MeanRisk(; obj = MaximumReturn(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(number_effective_assets(w), opt.nea, rtol = 1e-7)

        mre = MeanRisk(; obj = MaximumRatio(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(number_effective_assets(w), opt.nea, rtol = 5e-6)

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1,
                            tn = Turnover(; w = eqw, val = 0))
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w, eqw)
        @test all(abs.(w - eqw) .<= sqrt(eps()))

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1,
                            tn = Turnover(; w = eqw, val = 3e-2))
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        w = optimise!(mre, rd).w
        @test all(abs.(w - eqw) .- 3e-2 .<= sqrt(eps()))

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1,
                            tn = Turnover(; w = eqw, val = 1e-1))
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test all(abs.(w - eqw) .- 1e-1 .<= sqrt(eps()))

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1,
                            te = TrackingError(; err = 0,
                                               tracking = WeightsTracking(; w = eqw)))
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(norm(pr.X * (eqw - w), 2) / sqrt(T), 0, atol = 1e-10)

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1,
                            te = TrackingError(; err = 6e-3,
                                               tracking = WeightsTracking(; w = eqw)))
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        w = optimise!(mre, rd).w
        @test norm(pr.X * (w - eqw), 2) / sqrt(T) <= 6e-3

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1,
                            te = TrackingError(; err = 6e-3,
                                               tracking = WeightsTracking(; w = eqw)))
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test norm(pr.X * (w - eqw), 2) / sqrt(T) <= 6e-3

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1,
                            te = TrackingError(; formulation = NOCTracking(), err = 0,
                                               tracking = WeightsTracking(; w = eqw)))
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(norm(pr.X * (eqw - w), 1) / T, 0, atol = 1e-10)

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1,
                            te = TrackingError(; formulation = NOCTracking(), err = 6e-3,
                                               tracking = WeightsTracking(; w = eqw)))
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        w = optimise!(mre, rd).w
        @test norm(pr.X * (w - eqw), 1) / T <= 6e-3

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1,
                            te = TrackingError(; formulation = NOCTracking(), err = 6e-3,
                                               tracking = WeightsTracking(; w = eqw)))
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test norm(pr.X * (w - eqw), 1) / T <= 6e-3
    end
    @testset "Linear constraints" begin
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1)
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        w0 = optimise!(mre, rd).w

        lcs = [LinearConstraint(;
                                A = LinearConstraintSide(; group = :Assets, name = :ADI,
                                                         coef = 1), B = -0.05, comp = EQ()),
               LinearConstraint(;
                                A = LinearConstraintSide(; group = [:Assets, :Assets],
                                                         name = [:ADI, :TXN],
                                                         coef = [0.5, -1]), B = 0,
                                comp = LEQ()),
               LinearConstraint(;
                                A = LinearConstraintSide(;
                                                         group = [:Assets, :Assets,
                                                                  :Assets],
                                                         name = [:ADP, :AMGN, :MSFT],
                                                         coef = [1, 1, -1]), comp = GEQ(),
                                B = 0)]
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1,
                            lcs = lcs, sets = sets)
        mre = MeanRisk(; obj = MinimumRisk(), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w[rd.nx .== "ADI"][1], -0.05)
        @test 0.5 * w[rd.nx .== "ADI"][1] <= w[rd.nx .== "TXN"][1]
        @test w[rd.nx .== "ADP"][1] + w[rd.nx .== "AMGN"][1] >= w[rd.nx .== "MSFT"][1]

        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -1, ub = 1), sbgt = 1, bgt = 1,
                            lcs = linear_constraints(lcs, sets))
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        w = optimise!(mre, rd).w
        @test isapprox(w[rd.nx .== "ADI"][1], -0.05)
        @test 0.5 * w[rd.nx .== "ADI"][1] <= w[rd.nx .== "TXN"][1]
        @test w[rd.nx .== "ADP"][1] + w[rd.nx .== "AMGN"][1] >= w[rd.nx .== "MSFT"][1]
    end
    @testset "Philogeny constraints" begin
        plc = SemiDefinitePhilogenyConstraintEstimator(; pe = NetworkEstimator())
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                            sbgt = 0.2, cplg = plc)
        mre = MeanRisk(; r = StandardDeviation(), obj = MaximumRatio(; rf = rf), opt = opt)
        sol = optimise!(mre, rd)
        w = sol.w
        @test isapprox(w,
                       [-0.08941188587507842, 0.004087964849451435, -5.8729548108149055e-9,
                        4.075746090889191e-8, -0.0035874002038594243, 0.0015257322261556148,
                        0.0887650690884818, -0.01429783012008845, 0.08843777407332835,
                        0.0816429423100362, 0.16252779789133273, -0.007855304740591686,
                        1.6080811271032345e-8, 1.2883173429754384e-8, 0.004237398012094363,
                        0.027632579864173508, 0.02191355819332439, -2.5807518658931676e-8,
                        0.015567421726883687, 0.17623544582755232, 0.04473362305335808,
                        0.1253006525318998, 2.0807741637541986e-8, -0.045991869244797874,
                        0.031093507391791814, 0.20119149117297067, 3.491756541380375e-8,
                        0.06296984961556344, -6.294376604762954e-9, 0.023281394884115205],
                       rtol = 1e-6)

        plc = SemiDefinitePhilogenyConstraintEstimator(; pe = NetworkEstimator(), p = 0.5)
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                            sbgt = 0.2, cplg = plc)
        mre = MeanRisk(; r = StandardDeviation(), obj = MaximumRatio(; rf = rf), opt = opt)
        sol = optimise!(mre, rd)
        w = sol.w
        @test isapprox(w,
                       [-0.006486419523155919, 0.04157679040351318, 7.350857534505677e-9,
                        5.427979371115358e-8, 0.01969328111843075, 8.980510837920986e-9,
                        0.07073241900156549, 0.021107704903924867, 0.06392169581968478,
                        0.0636636242222088, 0.09066005748038097, 0.024681263808624433,
                        1.5109795596092596e-8, 9.222593072570602e-9, 0.03694940081813349,
                        0.045835435641328706, 2.135087670329354e-8, 1.4404484255586952e-9,
                        1.5646417946790892e-7, 0.11061524081435214, 0.07905751775733512,
                        0.07225614661607334, 6.856727872329261e-8, 5.612561391815739e-9,
                        0.0496735977706879, 0.10099270320804003, 2.8709884771856394e-8,
                        0.11506907666793889, 8.333511385906014e-9, 7.804864100089278e-8],
                       rtol = 1e-6)

        plc = SemiDefinitePhilogenyConstraintEstimator(; pe = NetworkEstimator())
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                            sbgt = 0.2, cplg = plc)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        sol = optimise!(mre, rd)
        w = sol.w
        @test isapprox(w,
                       [-0.08075934681743827, -3.649989141307891e-7, -1.841416023328978e-6,
                        1.4586316977894046e-6, -0.025356656487917302, 6.587821884291638e-6,
                        0.08086767826638101, -9.437095760182346e-6, 0.09229630268035095,
                        0.06658988527401016, 0.17344448834666254, 0.020624911550828802,
                        -1.2444862024537389e-6, 3.396661695142408e-7, -7.456244681296533e-7,
                        9.584973541705295e-7, 1.07045189896383e-5, -3.7788830159264194e-7,
                        0.06524449661197258, 0.1724857293358415, 0.03806377806175124,
                        0.18247929784910707, -1.0170194966113909e-7, -0.09383169952468239,
                        2.381027412207865e-6, 0.307841453911692, 5.139155038527589e-7,
                        6.112285629755843e-7, -4.497173200495211e-6, 4.7360186859669635e-6],
                       rtol = 1e-6)

        plc = SemiDefinitePhilogenyConstraintEstimator(; pe = NetworkEstimator(), p = 0.5)
        opt = JuMPOptimiser(; pe = pr, slv = slv,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                            sbgt = 0.2, cplg = plc)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        sol = optimise!(mre, rd)
        @test isequal(w, sol.w)

        plc = IntegerPhilogenyConstraintEstimator(; pe = NetworkEstimator())
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                            wb = WeightBoundsResult(; lb = -0.5, ub = 1), bgt = 1,
                            sbgt = 0.5, cplg = plc)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf),
                       opt = opt)
        sol = optimise!(mre, rd)
        w = sol.w
        @test isapprox(w,
                       [-0.0, -0.0, -0.0, -0.0, -0.011560495312185382, -0.0, -0.0, -0.0,
                        -0.0, -0.0, 0.4410620338641917, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
                        -0.0, -0.0, 1.0000000000000053, -0.0, -0.0, -0.0,
                        -0.4884395046878225, -0.0, 0.058937966135803016, -0.0, -0.0, -0.0,
                        -0.0], rtol = 1e-6)

        plc = IntegerPhilogenyConstraintEstimator(; pe = NetworkEstimator())
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                            wb = WeightBoundsResult(; lb = -0.2, ub = 1), bgt = 1,
                            sbgt = 0.5, nplg = plc)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf),
                       opt = opt)
        sol = optimise!(mre, rd)
        w = sol.w
        @test isapprox(w,
                       [-0.1, -0.0, -0.0, -0.0, -0.2, -0.0, -0.0, -0.0, -0.0, -0.0,
                        0.5292359178349958, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0,
                        0.6763228055813865, -0.0, -0.0, -0.0, -0.2, -0.0,
                        0.29444127658361774, -0.0, -0.0, -0.0, -0.0], rtol = 1e-6)

        plc = IntegerPhilogenyConstraintEstimator(; pe = NetworkEstimator())
        opt = JuMPOptimiser(; pe = pr, slv = mip_slv,
                            wb = WeightBoundsResult(; lb = 0, ub = 1), bgt = 1, sbgt = 0.5,
                            nplg = plc)
        mre = MeanRisk(; r = ConditionalValueatRisk(), obj = MaximumRatio(; rf = rf),
                       opt = opt)
        sol = optimise!(mre, rd)
        w = sol.w
        @test isapprox(w,
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.161686458243399,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.40016878959796814, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.4381447521586327, 0.0, 0.0, 0.0, 0.0],
                       rtol = 1e-6)

        ce = CentralityConstraintEstimator(; A = CentralityEstimator(), B = MinValue(),
                                           comp = LEQ())
        opt = JuMPOptimiser(; pe = pr, slv = slv, cent = ce, sets = sets)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test all(abs.(res.cent.A_ineq * w .- res.cent.B_ineq) .<= sqrt(eps()))
        @test isapprox(w,
                       [2.1447605251747904e-6, 4.9555522233308075e-6,
                        -3.048940921125532e-10, -6.335742240074599e-11,
                        3.0187478707549293e-6, 7.271579409066108e-10, 0.0005497745025639536,
                        3.263142400806418e-6, 3.221740106733761e-5, 7.303976772421854e-10,
                        0.19282060915575108, 4.763609364242417e-6, -6.554897009256369e-11,
                        -5.672390807110499e-11, 7.135752393246407e-10, 5.730381122392376e-6,
                        -3.104126163671757e-10, 6.88527743640993e-10, 7.100028708162024e-10,
                        0.28235947293416386, 0.03775974833251591, 0.07628255832555499,
                        7.493570890879621e-10, 7.441278087723419e-10, 6.505459842129682e-6,
                        0.41016523338701283, 7.246385613075014e-10, -4.4123969177698076e-11,
                        -3.1024091744633977e-10, -3.2453162827112235e-10], rtol = 1e-6)

        opt = JuMPOptimiser(; pe = pr, slv = slv, cent = ce, sets = sets,
                            wb = WeightBoundsResult(; lb = -0.2), sbgt = 0.2)
        mre = MeanRisk(; obj = MaximumRatio(; rf = rf), opt = opt)
        res = optimise!(mre, rd)
        w = res.w
        @test all(abs.(res.cent.A_ineq * w .- res.cent.B_ineq) .<= 10 * sqrt(eps()))
        @test isapprox(w,
                       [-0.04240425125756214, 1.0584787692400547e-7, -6.631057765173388e-7,
                        6.099347672589588e-7, -4.3112912853638063e-7, 1.6124012264484388e-7,
                        0.03150024923244586, -2.2747112277364104e-7, 5.900381877872145e-6,
                        1.2076023163590833e-6, 0.21355508452318192, 1.4262618788550154e-7,
                        -9.208897985580136e-9, 5.47082366787729e-6, -1.6733137292613466e-8,
                        1.5654863600976744e-7, 0.0025103099615115124, -0.10101441922708894,
                        3.2112528560829895e-7, 0.26334004450145576, 0.04529158679551895,
                        0.13272444153042973, 1.377444204938884e-7, -0.05657465192152953,
                        2.1202053830036573e-7, 0.3945871594431534, 0.08289991548144819,
                        0.03357215005600953, -7.881667397474365e-7, 9.080013146741573e-8],
                       rtol = 1e-6)
    end
end
